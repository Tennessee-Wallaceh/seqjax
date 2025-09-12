import typing
import time

from jaxtyping import PRNGKeyArray

import equinox as eqx
import jaxtyping
import jax.numpy as jnp
import jax.random as jrandom
import jax

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    BayesianSequentialModel,
)
from seqjax.model.typing import (
    HyperParametersType,
    InferenceParametersType,
)

from seqjax.inference.particlefilter import SMCSampler, run_filter, log_marginal
from seqjax.inference.mcmc.metropolis import (
    RandomWalkConfig,
    run_random_walk_metropolis,
)
from seqjax.inference.interface import inference_method
from seqjax import util


class ParticleMCMCConfig(
    eqx.Module,
    typing.Generic[ParticleType, ObservationType, ConditionType, ParametersType],
):
    """Configuration for :func:`run_particle_mcmc`."""

    particle_filter: SMCSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
    mcmc: RandomWalkConfig = RandomWalkConfig()
    initial_parameter_guesses: int = 10


@inference_method
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
    observation_path: ObservationType,
    config: ParticleMCMCConfig,
    *,
    condition_path: ConditionType | None = None,
    initial_latents: ParticleType | None = None,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_history: tuple[ObservationType, ...] | None = None,
    test_samples: int = 1000,
) -> tuple[jaxtyping.Array, InferenceParametersType, None]:
    """Sample parameters using particle marginal Metropolis-Hastings."""

    def estimate_log_joint(
        params: InferenceParametersType, key: PRNGKeyArray
    ) -> jaxtyping.Array:
        model_params = target_posterior.target_parameter(params)
        _, _, (log_marginal_increments,) = run_filter(
            config.particle_filter,
            key,
            model_params,
            observation_path,
            recorders=(log_marginal,),
        )
        log_prior = target_posterior.parameter_prior.log_prob(params, hyperparameters)
        return jnp.sum(log_marginal_increments) + log_prior

    init_time_start = time.time()
    init_key, sample_key = jrandom.split(key)
    initial_parameter_samples = jax.vmap(
        target_posterior.parameter_prior.sample, in_axes=[0, None]
    )(jrandom.split(init_key, config.initial_parameter_guesses), hyperparameters)
    parameter_init_marginals = jax.vmap(jax.jit(estimate_log_joint), in_axes=[0, None])(
        initial_parameter_samples,
        key,
    )
    init_time_end = time.time()
    init_time_s = init_time_end - init_time_start

    initial_parameters = typing.cast(
        InferenceParametersType,
        util.index_pytree(
            initial_parameter_samples, jnp.argmax(parameter_init_marginals).item()
        ),
    )

    sample_time_start = time.time()
    samples = typing.cast(
        InferenceParametersType,
        run_random_walk_metropolis(
            jax.jit(estimate_log_joint),
            sample_key,
            initial_parameters,
            config=config.mcmc,
            num_samples=test_samples,
        ),
    )
    sample_time_end = time.time()
    sample_time_s = sample_time_end - sample_time_start
    time_array_s = init_time_s + (
        jnp.arange(test_samples) * (sample_time_s / test_samples)
    )

    return time_array_s, samples, None
