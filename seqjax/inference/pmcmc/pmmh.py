from __future__ import annotations

from typing import TypeVar

import equinox as eqx
import jax
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
from seqjax.model.typing import Batched, SequenceAxis, HyperParametersType
from seqjax.inference.particlefilter import SMCSampler, run_filter


class RandomWalkConfig(eqx.Module):
    """Configuration for the random walk proposal used in PMCMC."""

    step_size: float = 0.1
    num_samples: int = 100


class ParticleMCMCConfig(eqx.Module):
    """Configuration for :func:`run_particle_mcmc`."""

    mcmc: RandomWalkConfig
    particle_filter: SMCSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]


Parameters = TypeVar("Parameters", bound=ParametersType)


def _propose_parameters(
    key: jrandom.PRNGKey,
    params: Parameters,
    step_size: float,
) -> Parameters:
    flat, unravel = jax.flatten_util.ravel_pytree(params)  # type: ignore[attr-defined]
    noise = step_size * jrandom.normal(key, flat.shape)
    return unravel(flat + noise)


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
) -> Batched[Parameters, SequenceAxis | int]:
    """Sample parameters using particle marginal Metropolis-Hastings."""

    pf = config.particle_filter
    mcmc_cfg = config.mcmc

    init_key, *step_keys = jrandom.split(key, mcmc_cfg.num_samples + 1)
    init_logp = _log_density(
        initial_parameters,
        init_key,
        pf,
        parameter_prior,
        hyper_parameters,
        observations,
        condition_path,
        initial_conditions,
        observation_history,
    )

    def step(state, rng):
        params, logp = state
        prop_key, pf_key, accept_key = jrandom.split(rng, 3)
        proposal = _propose_parameters(prop_key, params, mcmc_cfg.step_size)
        proposal_logp = _log_density(
            proposal,
            pf_key,
            pf,
            parameter_prior,
            hyper_parameters,
            observations,
            condition_path,
            initial_conditions,
            observation_history,
        )
        log_accept_ratio = proposal_logp - logp
        accept = jrandom.uniform(accept_key) < jnp.exp(log_accept_ratio)
        new_params = jax.tree_util.tree_map(
            lambda p, q: jnp.where(accept, q, p), params, proposal
        )
        new_logp = jnp.where(accept, proposal_logp, logp)
        return (new_params, new_logp), new_params

    _, samples = jax.lax.scan(
        step, (initial_parameters, init_logp), jnp.array(step_keys)
    )
    return samples
