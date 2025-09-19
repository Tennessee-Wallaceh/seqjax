from __future__ import annotations
from functools import partial

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


class NUTSConfig(eqx.Module):
    step_size: float = 1e-3
    num_adaptation: int = 1000
    num_warmup: int = 1000
    inverse_mass_matrix: Any | None = None
    num_chains: int = 1


@inference_method
def run_bayesian_nuts[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        ParticleT,
        InitialParticleT,
        TransitionParticleHistoryT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
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
) -> tuple[
    InferenceParametersT,
    tuple[jaxtyping.Array, ParticleT],
]:
    """Sample parameters and latent paths jointly using NUTS."""

    log_prob_joint = evaluate.get_log_prob_joint_for_target(target_posterior.target)

    def logdensity(state):
        latents, params = state
        log_prior = target_posterior.parameter_prior.log_prob(params, hyperparameters)
        model_params = target_posterior.target_parameter(params)
        log_like = log_prob_joint(
            latents, observation_path, condition_path, model_params
        )
        return log_like + log_prior

    def initial_state(key):
        param_key, latent_key = jrandom.split(key)
        initial_parameters = target_posterior.parameter_prior.sample(
            param_key, hyperparameters
        )
        initial_latents, _, _, _ = simulate(
            latent_key,
            target_posterior.target,
            None,
            target_posterior.target_parameter(initial_parameters),
            pytree_shape(observation_path)[0][0],
        )
        return (initial_latents, initial_parameters)

    warmup_key, init_key, sample_key = jrandom.split(key, 3)

    start_warmup_time = time.time()
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity, initial_step_size=config.step_size
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
    samples_per_chain = int(test_samples / config.num_chains)
    _, paths = inference_loop_multiple_chains(
        sample_key,
        nuts.step,
        warmup_states,
        num_samples=samples_per_chain,
        num_chains=config.num_chains,
    )
    param_samples: InferenceParametersT = jax.tree_util.tree_map(
        partial(jnp.concatenate, axis=-1), paths.position
    )[1]
    latent_samples: ParticleT = paths.position[0]

    end_time = time.time()
    sample_time = end_time - start_time
    time_array_s = warmup_time + (
        jnp.repeat(jnp.arange(samples_per_chain), config.num_chains)
        * (sample_time / samples_per_chain)
    )

    return param_samples, (time_array_s, latent_samples)
