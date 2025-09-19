import typing
from seqjax.inference.vi import transformed
from seqjax.inference.vi import base
from seqjax.inference.vi import transformations
from seqjax.inference import embedder
from seqjax.inference.vi import autoregressive
from seqjax.inference.vi import train
import optax  # type: ignore
import jax
import jax.numpy as jnp
from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    InferenceParametersType,
    BayesianSequentialModel,
)
from seqjax.model.typing import HyperParametersType
from seqjax.inference.interface import inference_method
import equinox as eqx
import jaxtyping
import jax.random as jrandom
from functools import partial
from dataclasses import field


class FullVIConfig(eqx.Module):
    learning_rate: float = 1e-2
    opt_steps: int = 5000
    parameter_field_bijections: dict[str, str | transformations.Bijector] = field(
        default_factory=dict
    )
    prev_window: int = 3
    post_window: int = 3
    observations_per_step: int = 10
    samples_per_context: int = 5


def get_interval_spline():
    return transformations.Chain(
        (
            transformations.Sigmoid(lower=-1.0, upper=1.0),
            transformations.ConstrainedRQS(num_bins=5, lower=-1.0, upper=1.0),
        )
    )


configured_bijections: dict[str, typing.Callable[[], transformations.Bijector]] = {
    "interval_spline": get_interval_spline,
    "sigmoid": partial(transformations.Sigmoid, lower=-1.0, upper=1.0),
}


class BufferedVIConfig(eqx.Module):
    learning_rate: float = 1e-2
    opt_steps: int = 5000
    parameter_field_bijections: dict[str, str] = field(default_factory=dict)
    buffer_length: int = 15
    batch_length: int = 10
    prev_window: int = 3
    post_window: int = 3
    observations_per_step: int = 10
    samples_per_context: int = 5
    nn_width: int = 20
    nn_depth: int = 2
    control_variate: bool = False


@inference_method
def run_full_path_vi(
    target_posterior: BayesianSequentialModel[
        ParticleType,
        ObservationType,
        ConditionType,
        ParametersType,
        InferenceParametersType,
        HyperParametersType,
    ],
    hyperparameters: HyperParametersType,
    key: jaxtyping.PRNGKeyArray,
    observation_path: ObservationType,
    condition_path: ConditionType | None = None,
    test_samples: int = 1000,
    config: FullVIConfig = FullVIConfig(),
) -> tuple[InferenceParametersType, typing.Any]:
    sequence_length = observation_path.batch_shape[0]
    y_dim = observation_path.flat_dim

    target_param_class = target_posterior.inference_parameter_cls
    target_latent_class = target_posterior.target.particle_cls

    # handle parameter constrainsts with specified constraint transforms
    field_bijections = {}
    for parameter_field, bijection in config.parameter_field_bijections.items():
        if isinstance(bijection, str):
            field_bijections[parameter_field] = configured_bijections[bijection]()
        else:
            field_bijections[parameter_field] = bijection

    parameter_approximation = transformed.transform_approximation(
        target_struct_class=target_param_class,
        base=base.MeanField,
        constraint=partial(
            transformations.FieldwiseBijector,
            field_bijections=field_bijections,
        ),
    )
    embed = embedder.WindowEmbedder(
        sequence_length, config.prev_window, config.post_window, y_dim
    )

    latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
        target_latent_class,
        buffer_length=0,
        batch_length=sequence_length,
        context_dim=embed.context_dimension + 1,  # just location
        parameter_dim=target_param_class.flat_dim,
        lag_order=1,
        nn_width=10,
        nn_depth=2,
        key=jrandom.key(10),
    )

    approximation: base.SSMVariationalApproximation = base.FullAutoregressiveVI(
        latent_approximation,
        parameter_approximation,
        embed,
    )

    optim = optax.apply_if_finite(
        optax.adam(config.learning_rate), max_consecutive_errors=100
    )
    run_tracker = train.DefaultTracker()
    fitted_approximation = train.train(
        model=approximation,
        observations=observation_path,
        conditions=condition_path,
        key=key,
        optim=optim,
        target=target_posterior,
        num_steps=config.opt_steps,
        run_tracker=run_tracker,
        observations_per_step=config.observations_per_step,
        samples_per_context=config.samples_per_context,
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
        (run_tracker,),
    )


@inference_method
def run_buffered_vi(
    target_posterior: BayesianSequentialModel[
        ParticleType,
        ObservationType,
        ConditionType,
        ParametersType,
        InferenceParametersType,
        HyperParametersType,
    ],
    hyperparameters: HyperParametersType,
    key: jaxtyping.PRNGKeyArray,
    observation_path: ObservationType,
    condition_path: ConditionType | None = None,
    test_samples: int = 1000,
    config: BufferedVIConfig = BufferedVIConfig(),
) -> tuple[InferenceParametersType, typing.Any]:
    sequence_length = observation_path.batch_shape[0]
    y_dim = observation_path.flat_dim

    key, subkey = jrandom.split(key)

    target_param_class = target_posterior.inference_parameter_cls
    target_latent_class = target_posterior.target.particle_cls

    parameter_approximation = transformed.transform_approximation(
        target_struct_class=target_param_class,
        base=base.MeanField,
        constraint=partial(
            transformations.FieldwiseBijector,
            field_bijections={
                field: configured_bijections[bijection_label]()
                for field, bijection_label in config.parameter_field_bijections.items()
            },
        ),
    )

    embed = embedder.WindowEmbedder(
        sequence_length, config.prev_window, config.post_window, y_dim
    )

    latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
        target_latent_class,
        buffer_length=config.buffer_length,
        batch_length=config.batch_length,
        context_dim=embed.context_dimension,
        parameter_dim=target_param_class.flat_dim,
        lag_order=1,
        nn_width=config.nn_width,
        nn_depth=config.nn_depth,
        key=jrandom.key(10),
    )

    approximation: base.SSMVariationalApproximation = base.BufferedSSMVI(
        latent_approximation,
        parameter_approximation,
        embed,
        control_variate=config.control_variate,
    )

    optim = optax.apply_if_finite(
        optax.adam(config.learning_rate), max_consecutive_errors=100
    )

    run_tracker = train.DefaultTracker()
    fitted_approximation = train.train(
        model=approximation,
        observations=observation_path,
        conditions=condition_path,
        key=key,
        optim=optim,
        target=target_posterior,
        num_steps=config.opt_steps,
        run_tracker=run_tracker,
        observations_per_step=config.observations_per_step,
        samples_per_context=config.samples_per_context,
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
        (approx_start, x_q, run_tracker),
    )
