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

from seqjax.inference.particlefilter import registry as particle_filter_registry
from seqjax.inference.particlefilter import SMCSampler, run_filter
from seqjax.inference.particlefilter.base import FilterData
from seqjax.inference.mcmc.metropolis import (
    RandomWalkConfig,
    run_random_walk_metropolis,
)
from seqjax.inference.interface import InferenceDataset, inference_method
from seqjax import util
import jax.scipy as jsp


def log_marginal_increment(filter_data: FilterData):
    return jax.lax.select(
        filter_data.ancestor_ix[0] == -1,
        jsp.special.logsumexp(filter_data.log_w) - jnp.log(filter_data.log_w.shape[0]),
        jsp.special.logsumexp(filter_data.resampled_log_w + filter_data.log_w_inc)
        - jsp.special.logsumexp(filter_data.resampled_log_w),
    )

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
    target_posterior: BayesianSequentialModel[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    hyperparameters: HyperParametersT,
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
                tuple[ParticleT, ...],
                ObservationT,
                ConditionT,
                ParametersT,
            ],
            sequence_key: PRNGKeyArray,
            params: InferenceParametersT,
            observation_path: ObservationT,
            condition_path: ConditionT | seqjtyping.NoCondition,
        ) -> jaxtyping.Array: ...

    def estimate_sequence_log_marginal(
        particle_filter: SMCSampler[
            ParticleT,
            tuple[ParticleT, ...],
            ObservationT,
            ConditionT,
            ParametersT,
        ],
        sequence_key: PRNGKeyArray,
        params: InferenceParametersT,
        observation_path: ObservationT,
        condition_path: ConditionT | seqjtyping.NoCondition,
    ) -> jaxtyping.Array:
        model_params = target_posterior.convert_to_model_parameters(params)
        _, _, (log_marginal_increments,) = run_filter(
            sequence_key,
            particle_filter,
            model_params,
            observation_path,
            condition_path=condition_path,
            recorders=(log_marginal_increment,),
        )
        return jnp.sum(log_marginal_increments)

    sequence_log_marginal_estimator: SequenceLogMarginalEstimator = (
        estimate_sequence_log_marginal
    )

    def estimate_log_joint(
        particle_filter: SMCSampler[
            ParticleT,
            tuple[ParticleT,...],
            ObservationT,
            ConditionT,
            ParametersT,
        ],
        params: InferenceParametersT,
        key: PRNGKeyArray,
    ) -> jaxtyping.Array:
        sequence_keys = jrandom.split(key, num_sequences)
        if isinstance(conditions, seqjtyping.NoCondition):
            log_marginal = jax.vmap(
                lambda sequence_key, observation_path: sequence_log_marginal_estimator(
                    particle_filter,
                    sequence_key,
                    params,
                    observation_path,
                    seqjtyping.NoCondition(),
                )
            )(sequence_keys, observations).sum()
        else:
            log_marginal = jax.vmap(
                lambda sequence_key, observation_path, condition_path: sequence_log_marginal_estimator(
                    particle_filter,
                    sequence_key,
                    params,
                    observation_path,
                    condition_path,
                )
            )(sequence_keys, observations, conditions).sum()

        log_prior = target_posterior.parameter_prior.log_prob(params, hyperparameters)
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
    dataset: InferenceDataset[ObservationT, ConditionT],
    test_samples: int,
    config: ParticleMCMCConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    """Sample parameters using particle marginal Metropolis-Hastings."""

    estimate_log_joint = _make_log_joint_estimator(
        target_posterior,
        hyperparameters,
        dataset,
    )

    particle_filter = particle_filter_registry._build_filter(
        target_posterior, 
        config=config.particle_filter_config
    )
    
    init_key, next_sample_key = jrandom.split(key)
    initial_parameters = target_posterior.parameter_prior.sample(init_key, hyperparameters)

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
        key, initial_parameters 
    ).compile()
    print("compiled")  
    print("steps:", config.num_steps)
    print("seconds:", config.time_limit_s)
    print("="*20)

    sample_blocks = [
        # add a leading batch axis
        jax.tree_util.tree_map(
            functools.partial(jnp.expand_dims, axis=0), 
            initial_parameters
        )
    ]
    samples_taken = 0
    block_times_s = []



    inference_time_start = time.time()
    while True:
        sample_key, next_sample_key = jrandom.split(next_sample_key)
        start_parameter = util.index_pytree(sample_blocks[-1], -1)
        samples = compiled_run(sample_key, start_parameter)
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
