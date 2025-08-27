import typing
from seqjax.inference.vi import transformed
from seqjax.inference.vi import base
from seqjax.inference.vi import transformations
from seqjax.inference.vi import embedder
from seqjax.inference.vi import autoregressive
from seqjax.inference.vi import train
import optax
import jax
import jax.numpy as jnp
import pandas as pd
from seqjax.model.simulate import simulate
from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    InferenceParametersType,
    SequentialModel,
    BayesianSequentialModel,
    ParameterPrior,
)
from seqjax.model.typing import Batched, SequenceAxis, SampleAxis, HyperParametersType
from seqjax.model import evaluate
from seqjax.util import pytree_shape
import equinox as eqx
import jaxtyping
import jax.random as jrandom
from functools import partial
from dataclasses import field


class FullVIConfig(eqx.Module):
    learning_rate: float = 1e-2
    opt_steps: int = 5000
    parameter_field_bijections: dict[str, transformations.Bijector] = field(
        default_factory=dict
    )
    metric_samples: int = 1000


class BufferedVIConfig(eqx.Module):
    learning_rate: float = 1e-2
    opt_steps: int = 5000
    parameter_field_bijections: dict[str, transformations.Bijector] = field(
        default_factory=dict
    )
    metric_samples: int = 1000
    buffer_length: int = 15
    batch_length: int = 10
    prev_window: int = 3
    post_window: int = 3


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
    observation_path: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    config: FullVIConfig = FullVIConfig(),
):

    sequence_length = observation_path.batch_shape[0]

    key, subkey = jrandom.split(key)
    initial_parameters = target_posterior.parameter_prior.sample(
        subkey, hyperparameters
    )
    initial_state = target_posterior.target.prior.sample(
        key, None, target_posterior.target_parameter(initial_parameters)
    )[0]

    target_param_class = type(initial_parameters)
    target_latent_class = type(initial_state)

    parameter_approximation = transformed.transform_approximation(
        target_struct_class=target_param_class,
        base=base.MeanField,
        constraint=partial(
            transformations.FieldwiseBijector,
            field_bijections=config.parameter_field_bijections,
        ),
    )

    embed = None  # no embed for full path vi

    latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
        target_latent_class,
        buffer_length=0,
        batch_length=sequence_length,
        context_dim=1,  # just location
        parameter_dim=target_param_class.flat_dim,
        lag_order=1,
        nn_width=10,
        nn_depth=2,
        key=jrandom.key(10),
    )

    approximation = base.FullAutoregressiveVI(
        latent_approximation,
        parameter_approximation,
        embed,
    )

    optim = optax.apply_if_finite(
        optax.adam(config.learning_rate), max_consecutive_errors=100
    )
    fitted_approximation, opt_state, run_tracker, elapsed_time_s = train.train(
        model=approximation,
        observations=observation_path,
        conditions=condition_path,
        key=key,
        optim=optim,
        target=target_posterior,
        num_steps=config.opt_steps,
        record_interval=100,
        observations_per_step=5,
        samples_per_context=10,
    )

    # run sample again for testing purposes
    n_context = 100
    s_per_context = int(config.metric_samples / n_context)
    theta_q, log_q_theta, x_q, log_q_x_path, start_ix, latent_scaling = (
        fitted_approximation.joint_sample_and_log_prob(
            observation_path, None, key, n_context, s_per_context
        )
    )

    flat_theta_q = jax.tree_util.tree_map(lambda x: jnp.ravel(x), theta_q)
    x_q = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, x.shape[-1])), x_q)
    run_data = pd.DataFrame(run_tracker.rows[1:])
    return (
        run_data.elapsed_time_s,
        x_q,
        flat_theta_q,
        run_data,
    )


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
    observation_path: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    config: BufferedVIConfig = BufferedVIConfig(),
):

    sequence_length = observation_path.batch_shape[0]
    y_dim = observation_path.flat_dim

    key, subkey = jrandom.split(key)
    initial_parameters = target_posterior.parameter_prior.sample(
        subkey, hyperparameters
    )
    initial_state = target_posterior.target.prior.sample(
        key, None, target_posterior.target_parameter(initial_parameters)
    )[0]

    target_param_class = type(initial_parameters)
    target_latent_class = type(initial_state)

    parameter_approximation = transformed.transform_approximation(
        target_struct_class=target_param_class,
        base=base.MeanField,
        constraint=partial(
            transformations.FieldwiseBijector,
            field_bijections=config.parameter_field_bijections,
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
        nn_width=10,
        nn_depth=2,
        key=jrandom.key(10),
    )

    approximation = base.BufferedSSMVI(
        latent_approximation,
        parameter_approximation,
        embed,
    )

    optim = optax.apply_if_finite(
        optax.adam(config.learning_rate), max_consecutive_errors=100
    )
    fitted_approximation, opt_state, run_tracker, elapsed_time_s = train.train(
        model=approximation,
        observations=observation_path,
        conditions=condition_path,
        key=key,
        optim=optim,
        target=target_posterior,
        num_steps=config.opt_steps,
        record_interval=100,
        observations_per_step=5,
        samples_per_context=10,
    )

    # run sample again for testing purposes
    n_context = 100
    s_per_context = int(config.metric_samples / n_context)
    theta_q, log_q_theta, x_q, log_q_x_path, start_ix, latent_scaling = (
        fitted_approximation.joint_sample_and_log_prob(
            observation_path, None, key, n_context, s_per_context
        )
    )

    flat_theta_q = jax.tree_util.tree_map(lambda x: jnp.ravel(x), theta_q)
    x_q = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, x.shape[-1])), x_q)
    run_data = pd.DataFrame(run_tracker.rows[1:])
    return (run_data.elapsed_time_s, x_q, flat_theta_q, (run_data, start_ix))
