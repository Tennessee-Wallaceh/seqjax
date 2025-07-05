from __future__ import annotations

from typing import Any, Tuple, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
    ParameterPrior,
)
from seqjax.model.typing import Batched, SequenceAxis, SampleAxis, HyperParametersType
from seqjax.model import evaluate

import blackjax  # type: ignore


class NUTSConfig(eqx.Module):
    """Configuration for :func:`run_nuts`."""

    step_size: float = 0.1
    num_warmup: int = 100
    num_samples: int = 100
    inverse_mass_matrix: Any | None = None


def run_nuts(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    observations: Batched[ObservationType, SequenceAxis],
    *,
    parameters: ParametersType,
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    initial_latents: Batched[ParticleType, SequenceAxis],
    config: NUTSConfig = NUTSConfig(),
    parameter_prior: ParameterPrior[ParametersType, HyperParametersType] | None = None,
    initial_parameters: ParametersType | None = None,
    hyper_parameters: HyperParametersType | None = None,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_history: tuple[ObservationType, ...] | None = None,
) -> Batched[ParticleType, SequenceAxis | int]:
    """Sample latent paths using the NUTS algorithm from ``blackjax``."""

    log_prob_joint = evaluate.get_log_prob_joint_for_target(target)

    def logdensity(x):
        return log_prob_joint(x, observations, condition_path, parameters)

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

    state, _ = jax.lax.scan(warmup_step, state, keys[: config.num_warmup])

    def sample_step(state, key):
        state, _ = nuts.step(key, state)
        return state, state.position

    _, samples = jax.lax.scan(sample_step, state, keys[config.num_warmup :])
    return samples


def run_bayesian_nuts(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    observations: Batched[ObservationType, SequenceAxis],
    *,
    parameter_prior: ParameterPrior[ParametersType, HyperParametersType],
    hyper_parameters: HyperParametersType | None = None,
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    initial_latents: Batched[ParticleType, SequenceAxis],
    initial_parameters: ParametersType,
    config: NUTSConfig = NUTSConfig(),
    parameters: ParametersType | None = None,
) -> Tuple[
    Batched[ParticleType, SequenceAxis | int],
    Batched[ParametersType, SampleAxis | int],
]:
    """Sample parameters and latent paths jointly using NUTS."""

    log_prob_joint = evaluate.get_log_prob_joint_for_target(target)

    def logdensity(state):
        latents, params = state
        log_prior = parameter_prior.log_prob(params, hyper_parameters)
        log_like = log_prob_joint(latents, observations, condition_path, params)
        return log_like + log_prior

    init_state = (initial_latents, initial_parameters)
    flat, _ = jax.flatten_util.ravel_pytree(init_state)  # type: ignore[attr-defined]
    dim = flat.shape[0]
    inv_mass = (
        jnp.ones(dim)
        if config.inverse_mass_matrix is None
        else config.inverse_mass_matrix
    )

    nuts = blackjax.nuts(
        logdensity, step_size=config.step_size, inverse_mass_matrix=inv_mass
    )
    state = nuts.init(init_state)

    keys = jax.random.split(key, config.num_warmup + config.num_samples)

    def warmup_step(state, key):
        state, _ = nuts.step(key, state)
        return state, None

    state, _ = jax.lax.scan(warmup_step, state, keys[: config.num_warmup])

    def sample_step(state, key):
        state, _ = nuts.step(key, state)
        return state, state.position

    _, samples = jax.lax.scan(sample_step, state, keys[config.num_warmup :])
    return samples


def run_nuts_parameters(
    logdensity: Callable[[ParametersType, PRNGKeyArray], jax.Array],
    key: PRNGKeyArray,
    initial_parameters: ParametersType,
    config: NUTSConfig = NUTSConfig(),
) -> Batched[ParametersType, SampleAxis | int]:
    """Sample parameters using the NUTS algorithm from ``blackjax``."""

    flat, _ = jax.flatten_util.ravel_pytree(initial_parameters)  # type: ignore[attr-defined]
    dim = flat.shape[0]
    inv_mass = (
        jnp.ones(dim)
        if config.inverse_mass_matrix is None
        else config.inverse_mass_matrix
    )

    nuts = blackjax.nuts(
        lambda p: logdensity(p, key),
        step_size=config.step_size,
        inverse_mass_matrix=inv_mass,
    )
    state = nuts.init(initial_parameters)

    keys = jax.random.split(key, config.num_warmup + config.num_samples)

    def warmup_step(state, key):
        state, _ = nuts.step(key, state)
        return state, None

    state, _ = jax.lax.scan(warmup_step, state, keys[: config.num_warmup])

    def sample_step(state, key):
        state, _ = nuts.step(key, state)
        return state, state.position

    _, samples = jax.lax.scan(sample_step, state, keys[config.num_warmup :])
    return samples
