from __future__ import annotations

from typing import TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    ParameterPrior,
    SequentialModel,
)
from seqjax.model.typing import Batched, SequenceAxis, SampleAxis, HyperParametersType
from functools import partial

from seqjax.inference.particlefilter import SMCSampler, run_filter
from seqjax.inference.mcmc.metropolis import (
    RandomWalkConfig,
    run_random_walk_metropolis,
)


class ParticleMCMCConfig(eqx.Module):
    """Configuration for :func:`run_particle_mcmc`."""

    mcmc: Callable[
        [
            Callable[[ParametersType, jrandom.PRNGKey], jnp.ndarray],
            jrandom.PRNGKey,
            ParametersType,
        ],
        Batched[ParametersType, SampleAxis | int],
    ] = partial(run_random_walk_metropolis, config=RandomWalkConfig())
    particle_filter: (
        SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType]
        | None
    ) = None


Parameters = TypeVar("Parameters", bound=ParametersType)


def _log_density(
    params: Parameters,
    key: jrandom.PRNGKey,
    pf: SMCSampler[ParticleType, ObservationType, ConditionType, Parameters],
    prior: ParameterPrior[Parameters, HyperParametersType],
    hyper_params: HyperParametersType | None,
    observations: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None,
    initial_conditions: tuple[ConditionType, ...] | None,
    observation_history: tuple[ObservationType, ...] | None,
) -> jnp.ndarray:
    _, _, log_mp, _, _ = run_filter(
        pf,
        key,
        params,
        observations,
        condition_path=condition_path,
        initial_conditions=initial_conditions,
        observation_history=observation_history,
    )
    log_like = log_mp[-1]
    log_prior = prior.log_prob(params, hyper_params)
    return log_like + log_prior


def run_particle_mcmc(
    target: SequentialModel[ParticleType, ObservationType, ConditionType, Parameters],
    key: jrandom.PRNGKey,
    observations: Batched[ObservationType, SequenceAxis],
    *,
    parameter_prior: ParameterPrior[Parameters, HyperParametersType],
    config: ParticleMCMCConfig,
    initial_parameters: Parameters,
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    hyper_parameters: HyperParametersType | None = None,
    initial_latents: Batched[ParticleType, SequenceAxis] | None = None,
    parameters: Parameters | None = None,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_history: tuple[ObservationType, ...] | None = None,
) -> Batched[Parameters, SampleAxis | int]:
    """Sample parameters using particle marginal Metropolis-Hastings."""

    pf = config.particle_filter
    if pf is None:
        raise ValueError("particle_filter must be provided in config")
    if pf.target is not target:
        pf = eqx.tree_at(lambda m: m.target, pf, target)
    sampler = config.mcmc

    def logdensity(params: Parameters, rng: jrandom.PRNGKey) -> jnp.ndarray:
        return _log_density(
            params,
            rng,
            pf,
            parameter_prior,
            hyper_parameters,
            observations,
            condition_path,
            initial_conditions,
            observation_history,
        )

    samples = sampler(logdensity, key, initial_parameters)
    return samples
