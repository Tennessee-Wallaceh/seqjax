"""Utilities for automatically tuning particle filters used in PMCMC."""

from __future__ import annotations

from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jaxtyping

import seqjax.model.typing as seqjtyping
from seqjax.model.base import BayesianSequentialModel
from seqjax.inference.particlefilter import SMCSampler
from seqjax import util


class ParticleFilterTuningConfig(eqx.Module):
    """Configuration for :func:`tune_particle_filter_variance`."""

    target_variance: float
    max_particles: int
    replications: int
    diagnostic_samples: int | None = None


LogJointEstimator = Callable[
    [
        SMCSampler,
        seqjtyping.Parameters,
        jaxtyping.PRNGKeyArray,
    ],
    jax.Array,
]


def tune_particle_filter_variance[
    ParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    estimate_log_joint: LogJointEstimator,
    base_filter: SMCSampler,
    target_posterior: BayesianSequentialModel,
    hyperparameters: HyperParametersT,
    config: ParticleFilterTuningConfig,
    key: jaxtyping.PRNGKeyArray,
) -> tuple[SMCSampler, dict[str, list[jax.Array]]]:
    """Heuristically tune the particle count of an :class:`SMCSampler`.

    The routine evaluates the variance of the log-marginal likelihood estimator
    for a set of prior-drawn parameter proposals across increasing particle
    counts until the empirical variance falls below ``config.target_variance``
    or the ``config.max_particles`` threshold is reached.

    Parameters
    ----------
    estimate_log_joint
        Callable implementing the log joint estimator used by the particle MCMC
        routine. The helper relies on this to avoid diverging implementations
        of the marginal likelihood estimator.
    base_filter
        The particle filter instance that will be cloned and tuned.
    target_posterior
        The sequential model describing the posterior of interest.
    hyperparameters
        Hyperparameters used when sampling parameters from the prior.
    config
        Tuning hyperparameters controlling the variance threshold, maximum
        number of particles, number of replications per parameter draw, and the
        number of diagnostic parameter draws.
    key
        PRNG key used for sampling diagnostic parameter proposals and running
        particle filters.

    Returns
    -------
    tuple
        The tuned :class:`SMCSampler` and a diagnostic log capturing the tested
        particle counts, per-parameter variances, and raw log-marginal samples
        produced during tuning.
    """

    if config.replications <= 0:
        raise ValueError("replications must be positive")

    parameter_draws = config.diagnostic_samples or 1
    if parameter_draws <= 0:
        raise ValueError("diagnostic_samples must be positive when provided")

    key, parameter_key = jrandom.split(key)
    parameter_keys = jrandom.split(parameter_key, parameter_draws)
    parameter_samples = jax.vmap(
        target_posterior.parameter_prior.sample, in_axes=[0, None]
    )(parameter_keys, hyperparameters)

    parameter_leaves = jax.tree_util.tree_leaves(parameter_samples)
    if not parameter_leaves:
        raise ValueError("parameter_samples produced an empty pytree")
    num_parameter_draws = parameter_leaves[0].shape[0]

    diagnostics: dict[str, list[jax.Array]] = {
        "particle_counts": [],
        "per_parameter_variance": [],
        "log_marginal_samples": [],
    }

    current_particles = int(base_filter.num_particles)
    tuned_filter = base_filter
    continue_search = True

    while continue_search:
        candidate_particles = min(current_particles, config.max_particles)
        candidate_filter = eqx.tree_at(
            lambda pf: pf.num_particles, base_filter, candidate_particles
        )

        per_parameter_logs: List[jax.Array] = []
        per_parameter_variances: List[jax.Array] = []

        for draw_ix in range(num_parameter_draws):
            params = util.index_pytree(parameter_samples, draw_ix)
            replicate_logs: List[jax.Array] = []

            for _ in range(config.replications):
                key, subkey = jrandom.split(key)
                log_joint = estimate_log_joint(candidate_filter, params, subkey)
                log_prior = target_posterior.parameter_prior.log_prob(
                    params, hyperparameters
                )
                log_marginal = log_joint - log_prior
                replicate_logs.append(log_marginal)

            log_samples = jnp.stack(replicate_logs)
            per_parameter_logs.append(log_samples)

            if config.replications > 1:
                variance = jnp.var(log_samples, ddof=1)
            else:
                variance = jnp.zeros_like(log_samples[0])
            per_parameter_variances.append(variance)

        log_marginal_matrix = jnp.stack(per_parameter_logs)
        variance_vector = jnp.stack(per_parameter_variances)

        diagnostics["particle_counts"].append(jnp.array(candidate_particles))
        diagnostics["log_marginal_samples"].append(log_marginal_matrix)
        diagnostics["per_parameter_variance"].append(variance_vector)

        tuned_filter = candidate_filter

        max_variance = float(jnp.max(variance_vector))
        if (max_variance <= config.target_variance) or (
            candidate_particles >= config.max_particles
        ):
            continue_search = False
        else:
            proposed_particles = candidate_particles * 2
            if proposed_particles > config.max_particles:
                proposed_particles = config.max_particles
            if proposed_particles == candidate_particles:
                continue_search = False
            else:
                current_particles = proposed_particles

    return tuned_filter, diagnostics

