"""
Run pulls together configuration to build and run variational models defined
in the registries.
"""

import typing

import jax
import jax.numpy as jnp
import jaxtyping
import optax  # type: ignore[import-untyped]

from seqjax.inference.vi import train
from seqjax.model.base import BayesianSequentialModel
import seqjax.model.typing as seqjtyping
from seqjax.inference.interface import InferenceDataset, inference_method
from seqjax.inference.vi import registry
from seqjax.inference.optimization import registry as optimization_registry

AdamOpt = optimization_registry.AdamOpt
AutoregressiveLatentApproximation = registry.AutoregressiveLatentApproximation


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
    dataset: InferenceDataset[ObservationT, ConditionT],
    test_samples: int,
    config: registry.FullVIConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    # set up a default tracker if none provided
    if tracker is None:
        tracker = train.Tracker(metric_samples=1000)

    observation_path, _ = dataset.sequence(0)
    sequence_length = observation_path.batch_shape[0]

    approximation = registry.build_approximation(
        config,
        sequence_length,
        target_posterior,
        key,
    )

    sync_interval_s = None

    if config.prior_training_optimization and not isinstance(
        config.prior_training_optimization, optimization_registry.NoOpt
    ):
        print("Starting prior-training...")
        prior_train_optim = optimization_registry.build_optimizer(
            config.prior_training_optimization
        )
        approximation, _ = train.train(
            model=approximation,
            dataset=dataset,
            key=key,
            optim=prior_train_optim,
            target=target_posterior,
            num_steps=config.prior_training_optimization.total_steps,
            run_tracker=tracker,
            sample_kwargs=config.training_sampling_kwargs(loss_label="param-prior"),
            loss_label="param-prior",
            time_limit_s=config.prior_training_optimization.time_limit_s,
            sync_interval_s=sync_interval_s,
        )

    if config.pre_training_optimization and not isinstance(
        config.pre_training_optimization, optimization_registry.NoOpt
    ):
        print("Starting pre-training...")
        pre_train_optim = optimization_registry.build_optimizer(
            config.pre_training_optimization
        )
        approximation, _ = train.train(
            model=approximation,
            dataset=dataset,
            key=key,
            optim=pre_train_optim,
            target=target_posterior,
            num_steps=config.pre_training_optimization.total_steps,
            run_tracker=tracker,
            sample_kwargs=config.training_sampling_kwargs(loss_label="pretrain"),
            loss_label="pretrain",
            time_limit_s=config.pre_training_optimization.time_limit_s,
            sync_interval_s=sync_interval_s,
        )

    opt_state: optax.GradientTransformation
    if isinstance(config.optimization, optimization_registry.NoOpt):
        fitted_approximation = approximation
        opt_state = None
    else:
        optim = optimization_registry.build_optimizer(config.optimization)

        fitted_approximation, opt_state = train.train(
            model=approximation,
            dataset=dataset,
            key=key,
            optim=optim,
            target=target_posterior,
            num_steps=config.optimization.total_steps,
            run_tracker=tracker,
            sample_kwargs=config.training_sampling_kwargs(loss_label="elbo"),
            time_limit_s=config.optimization.time_limit_s,
            sync_interval_s=sync_interval_s,
        )

    # run sample again for testing purposes
    eval_sampling_kwargs = config.evaluation_sampling_kwargs(test_samples=test_samples)
    theta_q, log_q_theta, x_q, log_q_x_path, _ = (
        fitted_approximation.joint_sample_and_log_prob(
            dataset,
            key,
            eval_sampling_kwargs,
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
    dataset: InferenceDataset[ObservationT, ConditionT],
    test_samples: int,
    config: registry.BufferedVIConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    # set up a default tracker if none provided
    if tracker is None:
        tracker = train.Tracker(metric_samples=5000)

    #TODO: find a way to pass dynamic objects to inference runs
    start_approximation = None
    sync_interval_s = None

    observation_path, _ = dataset.sequence(0)
    sequence_length = observation_path.batch_shape[0]

    if start_approximation is not None:
        approximation = start_approximation
    else:
        approximation = registry.build_approximation(
            config,
            sequence_length,
            target_posterior,
            key,
        )
    start_approximation = approximation
    
    opt_state: optax.GradientTransformation

    if config.prior_training_optimization and not isinstance(
        config.prior_training_optimization, optimization_registry.NoOpt
    ):
        print("Starting prior-training...")
        pre_train_optim = optimization_registry.build_optimizer(
            config.prior_training_optimization
        )
        approximation, _ = train.train(
            model=approximation,
            dataset=dataset,
            key=key,
            optim=pre_train_optim,
            target=target_posterior,
            num_steps=config.prior_training_optimization.total_steps,
            run_tracker=tracker,
            sample_kwargs=config.training_sampling_kwargs(loss_label="param-prior"),
            loss_label="param-prior",
            time_limit_s=config.prior_training_optimization.time_limit_s,
            sync_interval_s=sync_interval_s,
        )


    if config.pre_training_optimization and not isinstance(
        config.pre_training_optimization, optimization_registry.NoOpt
    ):
        print("Starting pre-training...")
        pre_train_optim = optimization_registry.build_optimizer(
            config.pre_training_optimization
        )
        approximation, _ = train.train(
            model=approximation,
            dataset=dataset,
            key=key,
            optim=pre_train_optim,
            target=target_posterior,
            num_steps=config.pre_training_optimization.total_steps,
            run_tracker=tracker,
            sample_kwargs=config.training_sampling_kwargs(loss_label="pretrain"),
            loss_label="pretrain",
            time_limit_s=config.pre_training_optimization.time_limit_s,
            sync_interval_s=sync_interval_s,
        )

    
    if isinstance(
        config.optimization, optimization_registry.NoOpt
    ) or config.optimization.total_steps == 0:
        fitted_approximation = approximation
        opt_state = None
    else:
        optim = optimization_registry.build_optimizer(config.optimization)
        fitted_approximation, opt_state = train.train(
            model=approximation,
            dataset=dataset,
            key=key,
            optim=optim,
            target=target_posterior,
            num_steps=config.optimization.total_steps,
            run_tracker=tracker,
            sample_kwargs=config.training_sampling_kwargs(loss_label="elbo"),
            time_limit_s=config.optimization.time_limit_s,
            sync_interval_s=sync_interval_s,
        )

    # run sample again for testing purposes
    eval_sampling_kwargs = config.evaluation_sampling_kwargs(test_samples=test_samples)
    (
        theta_q,
        log_q_theta,
        x_q,
        log_q_x_path,
        (approx_start, theta_mask, y_batch, c_batch, sequence_minibatch_rescaling),
    ) = fitted_approximation.joint_sample_and_log_prob(
        dataset,
        key,
        eval_sampling_kwargs,
    )

    flat_theta_q = jax.tree_util.tree_map(lambda x: jnp.ravel(x), theta_q)
    return (
        flat_theta_q,
        (
            approx_start,
            x_q,
            tracker,
            fitted_approximation,
            opt_state,
        ),
    )
