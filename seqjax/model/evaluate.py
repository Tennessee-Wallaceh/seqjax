"""Model evaluation utilities for computing log probabilities (method API)."""

import typing

import jax
import jax.numpy as jnp
from jaxtyping import Scalar

import seqjax.model.typing as seqjtyping
from seqjax.model.base import SequentialModelBase
from seqjax.util import index_pytree, slice_pytree


def slice_prior_conditions[
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
](
    condition: ConditionT,
    prior,
) -> tuple[typing.Any, ...]:
    """Slice prior-condition history from the full condition path."""
    return typing.cast(
        tuple[typing.Any, ...],
        tuple(index_pytree(condition, ix) for ix in range(prior.order)),
    )


def slice_emission_observation_history[
    ObservationT: seqjtyping.Observation,
](
    observation_path: ObservationT,
    emission,
) -> tuple[typing.Any, ...]:
    """Slice lagged observation history expected by emission."""
    sequence_start = emission.observation_dependency
    sequence_length = observation_path.batch_shape[0] - sequence_start
    return typing.cast(
        tuple[typing.Any, ...],
        tuple(
            slice_pytree(
                observation_path,
                sequence_start + i,
                sequence_start + i + sequence_length,
            )
            for i in range(-emission.observation_dependency, 0)
        ),
    )


def log_prob_x[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    target: SequentialModelBase[LatentT, ObservationT, ConditionT, ParametersT],
    x_path: LatentT,
    condition: ConditionT,
    parameters: ParametersT,
) -> Scalar:
    """Return ``log p(x)`` for method-based sequential model API."""

    sequence_start = target.prior_order - 1
    sequence_length = x_path.batch_shape[0] - sequence_start

    prior_latents = target.make_latent_context(
        *tuple(index_pytree(x_path, ix) for ix in range(target.prior_order))
    )
    prior_conditions = target.make_condition_context(
        *tuple(index_pytree(condition, ix) for ix in range(target.prior_order))
    )
    log_p_x = target.prior_log_prob(prior_latents, prior_conditions, parameters)

    latent_context = prior_latents

    for t in range(1, sequence_length):
        latent_index = sequence_start + t
        next_latent = index_pytree(x_path, latent_index)
        condition_t = index_pytree(condition, latent_index)
        log_p_x = log_p_x + target.transition_log_prob(
            latent_context,
            next_latent,
            condition_t,
            parameters,
        )
        latent_context = latent_context.append(next_latent)

    return log_p_x


def log_prob_y_given_x[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    target: SequentialModelBase[LatentT, ObservationT, ConditionT, ParametersT],
    x_path: LatentT,
    observation_path: ObservationT,
    condition: ConditionT,
    parameters: ParametersT,
    reference_emission: tuple[ObservationT, ...] | None = None,
) -> Scalar:
    """Return ``log p(y | x)`` for method-based sequential model API."""

    if reference_emission is None:
        reference_emission = ()
    elif len(reference_emission) != target.observation_dependency:
        raise jax.errors.JaxRuntimeError(
            "Reference emission must match observation_dependency"
        )

    sequence_start = target.prior_order - 1
    sequence_length = x_path.batch_shape[0] - sequence_start
    if observation_path.batch_shape[0] < sequence_length:
        raise jax.errors.JaxRuntimeError(
            f"observation_path length must be >= {sequence_length}, got {observation_path.batch_shape[0]}"
        )

    observation_context = target.make_observation_context(*reference_emission)
    log_p_y = jnp.array(0.0)

    for t in range(sequence_length):
        latent_index = sequence_start + t
        latent_values = tuple(
            index_pytree(x_path, latent_index - target.emission_order + 1 + i)
            for i in range(target.emission_order)
        )
        latent_context = target.make_latent_context(*latent_values)
        obs_t = index_pytree(observation_path, t)
        cond_t = index_pytree(condition, latent_index)
        log_p_y = log_p_y + target.emission_log_prob(
            latent_context,
            obs_t,
            observation_context,
            cond_t,
            parameters,
        )
        observation_context = observation_context.append(obs_t)

    return log_p_y


def log_prob_joint[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    target: SequentialModelBase[LatentT, ObservationT, ConditionT, ParametersT],
    x_path: LatentT,
    observation_path: ObservationT,
    condition: ConditionT,
    parameters: ParametersT,
    reference_emission: tuple[ObservationT, ...] | None = None,
) -> Scalar:
    """Return ``log p(x, y)`` for method-based sequential model API."""

    return log_prob_x(
        target,
        x_path,
        condition,
        parameters,
    ) + log_prob_y_given_x(
        target,
        x_path,
        observation_path,
        condition,
        parameters,
        reference_emission=reference_emission,
    )


log_prob_x_methods = log_prob_x
log_prob_y_given_x_methods = log_prob_y_given_x
log_prob_joint_methods = log_prob_joint
