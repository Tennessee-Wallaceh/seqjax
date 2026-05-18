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
from seqjax.model.interface import (
    BayesianSequentialModelProtocol,
)

from seqjax.inference.particlefilter import registry as particle_filter_registry
from seqjax.inference.particlefilter import SMCSampler, run_filter
from seqjax.inference.particlefilter.interface import FilterData
from seqjax.inference.mcmc.metropolis import (
    RandomWalkConfig,
    run_random_walk_metropolis,
)
from seqjax.inference.interface import InferenceDataset, inference_method
from seqjax import util


class ParticleMCMCConfig(
    eqx.Module,
):
    """Configuration for :func:`run_particle_mcmc`."""
    particle_filter_config: particle_filter_registry.BootstrapFilterConfig
    mcmc_config: RandomWalkConfig
    time_limit_s: None | float = None
    num_steps: None | int = 5000
    sample_block_size: int = 1000

def _make_log_joint_estimator[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    dataset: InferenceDataset[ObservationT, ConditionT],
):
    observations = dataset.observations
    conditions = dataset.conditions
    num_sequences = dataset.num_sequences

    class SequenceLogMarginalEstimator(typing.Protocol):
        def __call__(
            self,
            particle_filter: SMCSampler[
                ParticleT,
                ObservationT,
                ConditionT,
                ParametersT,
                InferenceParametersT,
            ],
            sequence_key: PRNGKeyArray,
            params: InferenceParametersT,
            observation_path: ObservationT,
            condition_path: ConditionT,
        ) -> jaxtyping.Array: ...

    def estimate_sequence_log_marginal(
        particle_filter: SMCSampler[
            ParticleT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
        ],
        sequence_key: PRNGKeyArray,
        inference_params: InferenceParametersT,
        observation_path: ObservationT,
        condition_path: ConditionT,
    ) -> jaxtyping.Array:
        _, _, (log_marginal_increments,) = run_filter(
            sequence_key,
            particle_filter,
            inference_params,
            observation_path,
            condition_path=condition_path,
            recorders=(lambda fd: fd.log_z_inc,),
        )
        return jnp.sum(log_marginal_increments)

    sequence_log_marginal_estimator: SequenceLogMarginalEstimator = (
        estimate_sequence_log_marginal
    )

    def estimate_log_joint(
        particle_filter:  SMCSampler[
            ParticleT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
        ],
        inference_params: InferenceParametersT,
        key: PRNGKeyArray,
    ) -> jaxtyping.Array:
        """
        Sequences are treated as IID. So vmap down seed per sequence.
        """
        sequence_keys = jrandom.split(key, num_sequences)
        if isinstance(conditions, seqjtyping.NoCondition):
            in_axes = (None, 0, None, 0, None)
        else:
            in_axes = (None, 0, None, 0, 0)

        log_marginal = jax.vmap(
            sequence_log_marginal_estimator,
            in_axes=in_axes
        )(
            particle_filter,
            sequence_keys,
            inference_params,
            observations,
            conditions,
        ).sum()
        log_prior = target_posterior.parameterization.log_prob(inference_params)
        return log_marginal + log_prior

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
    target_posterior: BayesianSequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    key: jaxtyping.PRNGKeyArray,
    dataset: InferenceDataset[ObservationT, ConditionT],
    test_samples: int,
    config: ParticleMCMCConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    """Sample parameters using particle marginal Metropolis-Hastings."""

    estimate_log_joint = _make_log_joint_estimator(
        target_posterior,
        dataset,
    )

    particle_filter = particle_filter_registry.build_filter(
        target_posterior, 
        config=config.particle_filter_config
    )

    print("="*10, "initializing...", "="*10)
    init_key, init_logp_key, next_sample_key = jrandom.split(key, 3)
    initial_parameters = target_posterior.parameterization.sample(init_key)
    initial_logp = jax.jit(functools.partial(estimate_log_joint, particle_filter))(initial_parameters, init_logp_key)
    print("complete.")

    # by default sample in chunks of 1000
    num_samples = config.sample_block_size

    print("="*10, " compiling... ", "="*10)
    compiled_run = jax.jit(
        functools.partial(
            run_random_walk_metropolis,
            logdensity=functools.partial(estimate_log_joint, particle_filter),
            config=config.mcmc_config,
            num_samples=num_samples,
        )
    ).lower(
        key, initial_parameters, initial_logp
    ).compile()
    print("compiled")  
    print("sample_block_size:", config.sample_block_size)
    print("max steps:", config.num_steps)
    print("seconds:", config.time_limit_s)
    print("="*20)

    sample_blocks = []
    samples_taken = 0
    block_times_s = []

    start_parameter = initial_parameters
    start_logp = initial_logp
    inference_time_start = time.time()
    while True:
        sample_key, next_sample_key = jrandom.split(next_sample_key)

        samples, (start_parameter, start_logp) = compiled_run(sample_key, start_parameter, start_logp)
        samples_taken += num_samples

        elapsed_time_s = time.time() - inference_time_start
        block_times_s.append((elapsed_time_s, samples_taken))
        sample_blocks.append(samples)

        if config.time_limit_s and elapsed_time_s > config.time_limit_s:
            print("Stopping due to time limit")
            break

        if config.num_steps and samples_taken >= config.num_steps:
            print("Stopping due to sample limit")
            break
        
        print(f"Elapsed time: {int(elapsed_time_s / 60)} minutes")

    all_samples = util.concat_pytree(*sample_blocks)
    return all_samples, block_times_s 
