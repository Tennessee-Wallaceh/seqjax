"""
Run pulls together configuration to build and run variational models defined
in the registries.
"""

import typing

import jax
import jax.numpy as jnp
import jaxtyping

from seqjax.inference.vi import train
from seqjax.model.base import BayesianSequentialModel
import seqjax.model.typing as seqjtyping
from seqjax.inference.interface import inference_method
from seqjax.inference.vi import registry
from seqjax.inference.optimization import registry as optimization_registry


@inference_method
def run_full_path_vi[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    hyperparameters: HyperParametersT,
    key: jaxtyping.PRNGKeyArray,
    observation_path: ObservationT,
    condition_path: ConditionT | None = None,
    test_samples: int = 1000,
    config: registry.FullVIConfig = registry.FullVIConfig(),
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    sequence_length = observation_path.batch_shape[0]

    approximation = registry.build_approximation(
        config,
        sequence_length,
        target_posterior,
        key,
    )

    optim = optimization_registry.build_optimizer(config.optimization)

    fitted_approximation, opt_state = train.train(
        model=approximation,
        observations=observation_path,
        conditions=condition_path,
        key=key,
        optim=optim,
        target=target_posterior,
        num_steps=config.optimization.total_steps,
        run_tracker=tracker,
        observations_per_step=config.observations_per_step,
        samples_per_context=config.samples_per_context,
        time_limit_s=config.optimization.time_limit_s,
    )

    # run sample again for testing purposes
    n_context = 100
    s_per_context = int(test_samples / n_context)
    theta_q, log_q_theta, x_q, log_q_x_path, _ = (
        fitted_approximation.joint_sample_and_log_prob(
            observation_path, None, key, n_context, s_per_context
        )
    )

    flat_theta_q = jax.tree_util.tree_map(lambda x: jnp.ravel(x), theta_q)
    return (
        flat_theta_q,
        (tracker, x_q, fitted_approximation, opt_state),
    )


@inference_method
def run_buffered_vi[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    hyperparameters: HyperParametersT,
    key: jaxtyping.PRNGKeyArray,
    observation_path: ObservationT,
    condition_path: ConditionT | None = None,
    test_samples: int = 1000,
    config: registry.BufferedVIConfig = registry.BufferedVIConfig(),
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    sequence_length = observation_path.batch_shape[0]

    approximation = registry.build_approximation(
        config,
        sequence_length,
        target_posterior,
        key,
    )

    optim = optimization_registry.build_optimizer(config.optimization)

    if config.pre_training_steps > 0:
        approximation, _ = train.train(
            model=approximation,
            observations=observation_path,
            conditions=condition_path,
            key=key,
            optim=optim,
            target=target_posterior,
            num_steps=config.pre_training_steps,
            run_tracker=tracker,
            observations_per_step=config.observations_per_step,
            samples_per_context=config.samples_per_context,
            pre_train=True,
        )

    fitted_approximation, opt_state = train.train(
        model=approximation,
        observations=observation_path,
        conditions=condition_path,
        key=key,
        optim=optim,
        target=target_posterior,
        num_steps=config.optimization.total_steps,
        run_tracker=tracker,
        observations_per_step=config.observations_per_step,
        samples_per_context=config.samples_per_context,
        time_limit_s=config.optimization.time_limit_s,
    )

    # run sample again for testing purposes
    n_context = 100
    s_per_context = int(test_samples / n_context)
    (
        theta_q,
        log_q_theta,
        x_q,
        log_q_x_path,
        (approx_start, theta_mask, y_batch, c_batch),
    ) = fitted_approximation.joint_sample_and_log_prob(
        observation_path, condition_path, key, n_context, s_per_context
    )

    flat_theta_q = jax.tree_util.tree_map(lambda x: jnp.ravel(x), theta_q)
    return (
        flat_theta_q,
        (approx_start, x_q, tracker, fitted_approximation, opt_state),
    )
