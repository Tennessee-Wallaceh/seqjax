from __future__ import annotations
from functools import partial
from dataclasses import dataclass, field
import typing
from typing import Any
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import jaxtyping
from jax_tqdm import scan_tqdm  # type: ignore[import-not-found]

from seqjax.model.simulate import simulate
from seqjax.model.base import (
    BayesianSequentialModel,
)
import seqjax.model.typing as seqjtyping
from seqjax.model import evaluate
from seqjax.util import pytree_shape
from seqjax.inference.interface import inference_method

import blackjax  # type: ignore


def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains
):
    @scan_tqdm(num_samples)
    def one_step(carry, inp):
        ix, states = carry
        _, rng = inp
        keys = jax.random.split(rng, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return (ix + 1, states), states

    keys = jax.random.split(rng_key, num_samples)
    final_state, states = jax.lax.scan(
        one_step, (0, initial_state), (jnp.arange(num_samples), keys)
    )

    return final_state[1], states


def subsample_posterior(samples, n_sub, key):
    if len(samples.shape) == 2:
        samples = jnp.expand_dims(samples, axis=-1)

    n_draws, n_chains, dim = samples.shape
    flat = samples.reshape(n_draws * n_chains, dim)
    n_total = flat.shape[0]
    if n_sub > n_total:
        raise ValueError("n_sub > total number of draws")

    idx = jrandom.choice(key, n_total, shape=(n_sub,), replace=False)
    return flat[idx, :]  # (n_sub, dim)

@dataclass
class NUTSConfig:
    step_size: float = 1e-3
    num_adaptation: int = 10000
    num_warmup: int = 1000
    num_steps: int = 5000  # length of the chain
    inverse_mass_matrix: Any | None = None
    num_chains: int = 1
    max_time_s: float | None = None

    initial_params: seqjtyping.Parameters | None = None
    initial_latents: seqjtyping.Latent | None = None

    def __post_init__(self):
        positive_fields = [
            "step_size", 
            "num_adaptation", 
            "num_warmup",
            "num_steps",
            "num_chains",
        ]
        for field in positive_fields:
            assert getattr(self, field) > 0, f"{field} must be > 0."

@inference_method
def run_bayesian_nuts[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    hyperparameters: HyperParametersT,
    key: jaxtyping.PRNGKeyArray,
    observation_path: ObservationT,
    condition_path: ConditionT,
    test_samples: int,
    config: NUTSConfig = NUTSConfig(),
    tracker: Any = None,
) -> tuple[
    InferenceParametersT,
    tuple[jaxtyping.Array, ParticleT, InferenceParametersT],
]:
    """Sample parameters and latent paths jointly using NUTS."""

    log_prob_joint = partial(
        evaluate.log_prob_joint,
        target_posterior.target,
    )

    def logdensity(state):
        latents, params = state
        log_prior = target_posterior.parameter_prior.log_prob(params, hyperparameters)
        model_params = target_posterior.convert_to_model_parameters(params)
        log_like = log_prob_joint(
            latents, observation_path, condition_path, model_params
        )
        return log_like + log_prior

    def initial_state(key):
        param_key, latent_key = jrandom.split(key)
        if config.initial_params is not None:
            initial_parameters = typing.cast(InferenceParametersT, config.initial_params)
        else:
            initial_parameters = target_posterior.parameter_prior.sample(
                param_key, hyperparameters
            )

        if config.initial_latents is not None:
            initial_latents = config.initial_latents
        else:
            initial_latents, _ = simulate(
                latent_key,
                target_posterior.target,
                target_posterior.convert_to_model_parameters(initial_parameters),
                pytree_shape(observation_path)[0][0],
                condition=condition_path,
            )
        return (initial_latents, initial_parameters)

    warmup_key, init_key, sample_key = jrandom.split(key, 3)

    start_warmup_time = time.time()
    warmup = blackjax.window_adaptation(
        blackjax.nuts, 
        logdensity, 
        initial_step_size=config.step_size
    )

    (_, parameters), _ = warmup.run(
        warmup_key,
        initial_state(init_key),
        num_steps=config.num_adaptation,
    )

    # configure with warmup params
    nuts = blackjax.nuts(logdensity, **parameters)

    chain_inits = jax.vmap(initial_state)(jrandom.split(key, config.num_chains))
    initial_states = jax.vmap(nuts.init, in_axes=(0))(chain_inits)

    warmup_states, _ = inference_loop_multiple_chains(
        sample_key,
        nuts.step,
        initial_states,
        num_samples=config.num_warmup,
        num_chains=config.num_chains,
    )
    jax.block_until_ready(warmup_states)
    end_warmup_time = time.time()
    warmup_time = end_warmup_time - start_warmup_time

    start_time = time.time()
    samples_per_chain = int(config.num_steps / config.num_chains)
    _, paths = inference_loop_multiple_chains(
        sample_key,
        nuts.step,
        warmup_states,
        num_samples=samples_per_chain,
        num_chains=config.num_chains,
    )

    end_time = time.time()
    sample_time = end_time - start_time
    time_array_s = warmup_time + (
        jnp.repeat(jnp.arange(samples_per_chain), config.num_chains)
        * (sample_time / samples_per_chain)
    )

    # randomly down sample chains to give desired number of test samples
    latent_samples: ParticleT = paths.position[0]
    full_param_samples: InferenceParametersT = paths.position[1]
    param_samples: InferenceParametersT = jax.tree_util.tree_map(
        partial(subsample_posterior, n_sub=test_samples, key=key), full_param_samples
    )

    return jax.tree_util.tree_map(jnp.squeeze, param_samples), (
        time_array_s,
        latent_samples,
        full_param_samples,
    )
