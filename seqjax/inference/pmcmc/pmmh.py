from typing import TypeVar, Callable

from jaxtyping import PRNGKeyArray

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    ParameterPrior,
    SequentialModel,
    BayesianSequentialModel,
)
from seqjax.model.typing import (
    Batched,
    SequenceAxis,
    SampleAxis,
    HyperParametersType,
    InferenceParametersType,
)
from functools import partial

from seqjax.inference.particlefilter import SMCSampler, run_filter, log_marginal
from seqjax.inference.mcmc.metropolis import (
    RandomWalkConfig,
    run_random_walk_metropolis,
)
from seqjax import util


class ParticleMCMCConfig(eqx.Module):
    """Configuration for :func:`run_particle_mcmc`."""

    mcmc: Callable[
        [
            Callable[[ParametersType, jrandom.PRNGKey], jnp.ndarray],
            jrandom.PRNGKey,
            ParametersType,
        ],
        Batched[ParametersType, SampleAxis | int],
    ]
    particle_filter: SMCSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
    initial_parameter_guesses: int = 10


def run_particle_mcmc(
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
    config: ParticleMCMCConfig,
    observation_path: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
) -> Batched[ParametersType, SampleAxis]:
    """Sample parameters using particle marginal Metropolis-Hastings."""

    def estimate_log_joint(params, key):
        model_params = target_posterior.target_parameter(params)
        _, _, _, (log_marginal_increments,) = run_filter(
            config.particle_filter,
            key,
            model_params,
            observation_path,
            recorders=(log_marginal(),),
        )
        log_prior = target_posterior.parameter_prior.log_prob(params, hyperparameters)
        return jnp.sum(log_marginal_increments) + log_prior

    init_key, sample_key = jrandom.split(key)
    initial_parameter_samples = jax.vmap(
        target_posterior.parameter_prior.sample, in_axes=[0, None]
    )(jrandom.split(init_key, config.initial_parameter_guesses), hyperparameters)
    parameter_init_marginals = jax.vmap(jax.jit(estimate_log_joint), in_axes=[0, None])(
        initial_parameter_samples,
        key,
    )

    initial_parameters = util.index_pytree(
        initial_parameter_samples, jnp.argmax(parameter_init_marginals).item()
    )

    samples = run_random_walk_metropolis(
        estimate_log_joint, sample_key, initial_parameters, config=config.mcmc
    )
    return None, samples
