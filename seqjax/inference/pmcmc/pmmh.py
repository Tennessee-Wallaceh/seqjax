import functools
import typing
import time

from jaxtyping import PRNGKeyArray

import equinox as eqx
import jaxtyping
import jax.numpy as jnp
import jax.random as jrandom
import jax
import seqjax.model.typing as seqjtyping
from seqjax.model.base import (
    BayesianSequentialModel,
)

from seqjax.inference.particlefilter import SMCSampler, run_filter
from seqjax.inference.particlefilter.base import FilterData
from seqjax.inference.mcmc.metropolis import (
    RandomWalkConfig,
    run_random_walk_metropolis,
)
from seqjax.inference.interface import inference_method
from seqjax import util
from .tuning import (
    ParticleFilterTuningConfig,
    tune_particle_filter_variance,
)
import jax.scipy as jsp


def log_marginal_increment(filter_data: FilterData):
    return jax.lax.select(
        filter_data.ancestor_ix[0] == -1,
        jsp.special.logsumexp(filter_data.log_w) - jnp.log(filter_data.log_w.shape[0]),
        jsp.special.logsumexp(filter_data.resampled_log_w + filter_data.log_w)
        - jsp.special.logsumexp(filter_data.resampled_log_w),
    )


class ParticleMCMCConfig(
    eqx.Module,
):
    """Configuration for :func:`run_particle_mcmc`."""

    particle_filter: SMCSampler
    mcmc: RandomWalkConfig = RandomWalkConfig()
    initial_parameter_guesses: int = 10
    tuning: ParticleFilterTuningConfig | None = None


def _make_log_joint_estimator[
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
    observation_path: ObservationT,
    condition_path: ConditionT | None,
):
    def estimate_log_joint(
        particle_filter: SMCSampler[
            ParticleT,
            ObservationT,
            ConditionT,
            ParametersT,
        ],
        params: InferenceParametersT,
        key: PRNGKeyArray,
    ) -> jaxtyping.Array:
        _, _, (log_marginal_increments,) = run_filter(
            key,
            particle_filter,
            params,
            observation_path,
            condition_path=seqjtyping.NoCondition(),
            recorders=(log_marginal_increment,),
        )
        log_prior = target_posterior.parameter_prior.log_prob(params, hyperparameters)
        return jnp.sum(log_marginal_increments) + log_prior

    return estimate_log_joint


@inference_method
def run_particle_mcmc[
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
    config: ParticleMCMCConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, tuple[jaxtyping.Array]]:
    """Sample parameters using particle marginal Metropolis-Hastings."""

    estimate_log_joint = _make_log_joint_estimator(
        target_posterior, hyperparameters, observation_path, condition_path
    )

    tuning_diagnostics = None
    working_config = config
    if config.tuning is not None:
        key, tuning_key = jrandom.split(key)
        tuned_filter, tuning_diagnostics = tune_particle_filter_variance(
            estimate_log_joint,
            config.particle_filter,
            target_posterior,
            hyperparameters,
            config.tuning,
            tuning_key,
        )
        working_config = eqx.tree_at(
            lambda c: c.particle_filter,
            config,
            tuned_filter,
        )

    particle_filter = working_config.particle_filter
    log_joint_fn = functools.partial(estimate_log_joint, particle_filter)
    jit_log_joint_fn = jax.jit(log_joint_fn)

    init_time_start = time.time()
    init_key, sample_key = jrandom.split(key)
    initial_parameter_samples = jax.vmap(
        target_posterior.parameter_prior.sample, in_axes=[0, None]
    )(jrandom.split(init_key, config.initial_parameter_guesses), hyperparameters)
    parameter_init_marginals = jax.vmap(jit_log_joint_fn, in_axes=[0, None])(
        initial_parameter_samples,
        key,
    )
    init_time_end = time.time()
    init_time_s = init_time_end - init_time_start

    initial_parameters = typing.cast(
        InferenceParametersT,
        util.index_pytree(
            initial_parameter_samples, jnp.argmax(parameter_init_marginals).item()
        ),
    )

    sample_time_start = time.time()
    samples = typing.cast(
        InferenceParametersT,
        run_random_walk_metropolis(
            jit_log_joint_fn,
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

    diagnostics: list[typing.Any] = [time_array_s]
    if tuning_diagnostics is not None:
        diagnostics.append(tuning_diagnostics)

    return samples, tuple(diagnostics)
