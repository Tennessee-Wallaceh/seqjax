"""Model evaluation utilities for computing log probabilities."""

import typing

import jax
from jaxtyping import Scalar

import seqjax.model.typing as seqjtyping
from seqjax import util
from seqjax.model import interface as model_interface
from seqjax.model import util as model_util


def _validate_x_sequence_lengths[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        typing.Any,
    ],
    x_path: LatentT,
    condition: ConditionT,
) -> int:
    x_length = x_path.batch_shape[0]
    if x_length < target.prior_order:
        raise ValueError(
            "x_path length must be >= prior_order, got "
            f"x_length={x_length} prior_order={target.prior_order}"
        )

    sequence_length = x_length - target.prior_order + 1

    if not isinstance(condition, seqjtyping.NoCondition):
        condition_length = condition.batch_shape[0]
        min_condition_length = target.prior_order + sequence_length - 1
        if condition_length < min_condition_length:
            raise ValueError(
                "condition length is too short for latent evaluation, got "
                f"condition_length={condition_length} expected_at_least={min_condition_length}"
            )

    return sequence_length


def _validate_xy_sequence_lengths[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        typing.Any,
    ],
    x_path: LatentT,
    observation_path: ObservationT,
    condition: ConditionT,
) -> int:
    sequence_length = _validate_x_sequence_lengths(target, x_path, condition)

    y_length = observation_path.batch_shape[0]
    min_y_length = target.observation_dependency + sequence_length
    if y_length < min_y_length:
        raise ValueError(
            "observation_path length is too short for model dependency, got "
            f"y_length={y_length} expected_at_least={min_y_length} "
            f"(observation_dependency={target.observation_dependency}, "
            f"sequence_length={sequence_length})"
        )

    if not isinstance(condition, seqjtyping.NoCondition):
        condition_length = condition.batch_shape[0]
        min_condition_length = target.observation_dependency + sequence_length
        if condition_length < min_condition_length:
            raise ValueError(
                "condition length is too short for observation evaluation, got "
                f"condition_length={condition_length} expected_at_least={min_condition_length}"
            )

    return sequence_length


def _batched_latent_history[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    x_path: LatentT,
    order: int,
    sequence_length: int,
) -> model_interface.LatentContext[LatentT]:

    return target.latent_context(
        tuple(
            util.slice_pytree(
                x_path,
                target.prior_order + lag,
                target.prior_order + lag + sequence_length,
            )
            for lag in range(-order, 0)
        )
    )


def _batched_observation_history[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    observation_path: ObservationT,
    sequence_length: int,
) -> model_interface.ObservationContext[ObservationT]:
    dependency = target.observation_dependency
    return target.observation_context(
        tuple(
            util.slice_pytree(
                observation_path,
                dependency + lag,
                dependency + lag + sequence_length,
            )
            for lag in range(-dependency, 0)
        )
    )


def log_prob_x[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    x_path: LatentT,
    condition: ConditionT,
    parameters: ParametersT,
) -> Scalar:
    """Return ``log p(x)`` for a latent sequence."""
    sequence_length = _validate_x_sequence_lengths(target, x_path, condition)

    if len(parameters.batch_shape) == 0:
        parameters_batched = util.broadcast_packable(
            parameters,
            leading_axis_len=sequence_length,
        )
    else:
        parameters_batched = parameters

    prior_latent = target.latent_context(
        tuple(util.index_pytree(x_path, ix) for ix in range(target.prior_order))
    )
    prior_condition = model_util.slice_prior_context(target, condition)
    prior_log_p = target.prior_log_prob(
        prior_latent,
        prior_condition,
        util.index_pytree(parameters_batched, 0),
    )

    transition_steps = sequence_length - 1
    if transition_steps == 0:
        return prior_log_p

    transition_history = _batched_latent_history(
        target,
        x_path,
        target.transition_order,
        transition_steps,
    )
    transition_latent = util.slice_pytree(
        x_path,
        target.prior_order,
        target.prior_order + transition_steps,
    )
    transition_parameters = util.slice_pytree(parameters_batched, 1, sequence_length)

    if isinstance(condition, seqjtyping.NoCondition):
        transition_log_ps = jax.vmap(
            lambda latent_history_t, latent_t, params_t: target.transition_log_prob(
                latent_history_t,
                latent_t,
                condition,
                params_t,
            )
        )(
            transition_history,
            transition_latent,
            transition_parameters,
        )
    else:
        transition_condition = util.slice_pytree(
            condition,
            target.prior_order,
            target.prior_order + transition_steps,
        )
        transition_log_ps = jax.vmap(target.transition_log_prob)(
            transition_history,
            transition_latent,
            transition_condition,
            transition_parameters,
        )

    return prior_log_p + transition_log_ps.sum()


def log_prob_y_given_x[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    x_path: LatentT,
    observation_path: ObservationT,
    condition: ConditionT,
    parameters: ParametersT,
) -> Scalar:
    """Return ``log p(y | x)`` for a sequence of observations."""
    sequence_length = _validate_xy_sequence_lengths(
        target,
        x_path,
        observation_path,
        condition,
    )

    if len(parameters.batch_shape) == 0:
        parameters_batched = util.broadcast_packable(
            parameters,
            leading_axis_len=sequence_length,
        )
    else:
        parameters_batched = parameters

    emission_latent_history = _batched_latent_history(
        target,
        x_path,
        target.emission_order,
        sequence_length,
    )

    observation_start = target.observation_dependency
    observations = util.slice_pytree(
        observation_path,
        observation_start,
        observation_start + sequence_length,
    )
    emission_observation_history = _batched_observation_history(
        target,
        observation_path,
        sequence_length,
    )

    if isinstance(condition, seqjtyping.NoCondition):
        emission_log_ps = jax.vmap(
            lambda latent_history_t, observation_t, observation_history_t, params_t: target.emission_log_prob(
                latent_history_t,
                observation_t,
                observation_history_t,
                condition,
                params_t,
            )
        )(
            emission_latent_history,
            observations,
            emission_observation_history,
            parameters_batched,
        )
    else:
        observation_condition = util.slice_pytree(
            condition,
            observation_start,
            observation_start + sequence_length,
        )
        emission_log_ps = jax.vmap(target.emission_log_prob)(
            emission_latent_history,
            observations,
            emission_observation_history,
            observation_condition,
            parameters_batched,
        )
    return emission_log_ps.sum()


def log_prob_joint[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    x_path: LatentT,
    observation_path: ObservationT,
    condition: ConditionT,
    parameters: ParametersT,
) -> Scalar:
    """Return ``log p(x, y)`` for a path and observations."""
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
    )
