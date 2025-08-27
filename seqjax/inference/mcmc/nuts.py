from __future__ import annotations
from functools import partial

from typing import Any, Tuple, Callable
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jaxtyping import PRNGKeyArray

from seqjax.model.simulate import simulate
from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    InferenceParametersType,
    SequentialModel,
    BayesianSequentialModel,
    ParameterPrior,
)
from seqjax.model.typing import Batched, SequenceAxis, SampleAxis, HyperParametersType
from seqjax.model import evaluate
from seqjax.util import pytree_shape

import blackjax  # type: ignore


def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains
):

    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states

    keys = jax.random.split(rng_key, num_samples)
    final_state, states = jax.lax.scan(one_step, initial_state, keys)

    return final_state, states


class NUTSConfig(eqx.Module):
    step_size: float = 1e-3
    num_adaptation: int = 1000
    num_warmup: int = 1000
    inverse_mass_matrix: Any | None = None
    num_chains: int = 1


def run_latent_nuts(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    observation_path: Batched[ObservationType, SequenceAxis],
    parameters: ParametersType,
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    *,
    config: NUTSConfig = NUTSConfig(),
    initial_latents: Batched[ParticleType, SequenceAxis] | None = None,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_history: tuple[ObservationType, ...] | None = None,
) -> Batched[ParticleType, SampleAxis, SequenceAxis]:
    """Sample latent paths using the NUTS algorithm from ``blackjax``."""

    log_prob_joint = evaluate.get_log_prob_joint_for_target(target)

    def logdensity(x):
        return log_prob_joint(x, observation_path, condition_path, parameters)

    if initial_latents is None:
        key, latent_key = jrandom.split(key)
        initial_latents, _, _, _ = simulate(
            latent_key,
            target,
            condition_path,
            parameters,
            pytree_shape(observation_path)[0][0],
        )

    flat, _ = jax.flatten_util.ravel_pytree(initial_latents)  # type: ignore[attr-defined]
    dim = flat.shape[0]
    inv_mass = (
        jnp.ones(dim)
        if config.inverse_mass_matrix is None
        else config.inverse_mass_matrix
    )

    nuts = blackjax.nuts(
        logdensity, step_size=config.step_size, inverse_mass_matrix=inv_mass
    )
    state = nuts.init(initial_latents)

    keys = jax.random.split(key, config.num_warmup + config.num_samples)

    def warmup_step(state, key):
        state, _ = nuts.step(key, state)
        return state, None

    start_warmup_time = time.time()
    state, _ = jax.lax.scan(warmup_step, state, keys[: config.num_warmup])
    jax.block_until_ready(state)
    end_warmup_time = time.time()
    warmup_time = end_warmup_time - start_warmup_time

    def sample_step(state, key):
        state, _ = nuts.step(key, state)
        return state, state.position

    start_time = time.time()
    _, samples = jax.lax.scan(sample_step, state, keys[config.num_warmup :])
    jax.block_until_ready(samples)
    end_time = time.time()
    sample_time = end_time - start_time
    time_array_s = (
        warmup_time
        + jnp.ones(config.num_samples) * (sample_time / config.num_samples) * 60
    )
    return time_array_s, samples


def run_bayesian_nuts(
    target_posterior: BayesianSequentialModel[
        ParticleType,
        ObservationType,
        ConditionType,
        ParametersType,
        InferenceParametersType,
        HyperParametersType,
    ],
    hyperparameters: HyperParametersType,
    key: PRNGKeyArray,
    observation_path: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    config: NUTSConfig = NUTSConfig(),
    test_samples: int = 1000,
) -> Tuple[
    Batched[ParticleType, SampleAxis, SequenceAxis | int],
    Batched[ParametersType, SampleAxis | int],
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
    latent_samples, param_samples = jax.tree_util.tree_map(
        partial(jnp.concatenate, axis=-1), paths.position
    )

    end_time = time.time()
    sample_time = end_time - start_time
    time_array_s = warmup_time + (
        jnp.repeat(jnp.arange(samples_per_chain), config.num_chains)
        * (sample_time / samples_per_chain)
    )

    return time_array_s, latent_samples, param_samples, None
