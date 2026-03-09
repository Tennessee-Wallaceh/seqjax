"""Model evaluation utilities for computing log probabilities."""

import typing

import jax
import jax.numpy as jnp
from jaxtyping import Scalar

import seqjax.model.typing as seqjtyping
from seqjax import util
from seqjax.model import interface as model_interface
from seqjax.model import util as model_util


def _validate_sequence_lengths[
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
    observation_path: ObservationT | None,
    condition: ConditionT,
) -> int:
    x_length = x_path.batch_shape[0]
    if x_length < target.prior_order:
        raise ValueError(
            "x_path length must be >= prior_order, got "
            f"x_length={x_length} prior_order={target.prior_order}"
        )

    sequence_length = x_length - target.prior_order + 1

    if observation_path is not None:
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
        if condition_length < target.prior_order:
            raise ValueError(
                "condition length must be >= prior_order, got "
                f"condition_length={condition_length} prior_order={target.prior_order}"
            )

        if observation_path is None:
            min_condition_length = target.prior_order + sequence_length - 1
        else:
            min_condition_length = target.observation_dependency + sequence_length

        if condition_length < min_condition_length:
            raise ValueError(
                "condition length is too short for requested evaluation, got "
                f"condition_length={condition_length} expected_at_least={min_condition_length}"
            )

    return sequence_length


def _latent_context_at[
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
    end_index: int,
    order: int,
) -> model_interface.LatentContext[LatentT]:
    start_index = end_index - order + 1
    return target.latent_context(
        tuple(
            util.dynamic_index_pytree_in_dim(x_path, start_index + offset, 0)
            for offset in range(order)
        )
    )


def _observation_context_at[
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
    end_index: int,
) -> model_interface.ObservationContext[ObservationT]:
    order = target.observation_dependency
    start_index = end_index - order + 1
    return target.observation_context(
        tuple(
            util.dynamic_index_pytree_in_dim(observation_path, start_index + offset, 0)
            for offset in range(order)
        )
    )


def _condition_at[ConditionT: seqjtyping.Condition](
    condition: ConditionT,
    index: int,
) -> ConditionT:
    if isinstance(condition, seqjtyping.NoCondition):
        return typing.cast(ConditionT, condition)
    return util.dynamic_index_pytree_in_dim(condition, index, 0)


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
    sequence_length = _validate_sequence_lengths(target, x_path, None, condition)

    parameters_batched = util.broadcast_packable(
        parameters,
        leading_axis_len=sequence_length,
    )

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

    def scan_transition(_, step_index: jax.Array) -> tuple[None, Scalar]:
        target_index = target.prior_order + step_index
        latent_history = _latent_context_at(
            target,
            x_path,
            end_index=target_index - 1,
            order=target.transition_order,
        )
        latent = util.dynamic_index_pytree_in_dim(x_path, target_index, 0)
        transition_condition = _condition_at(condition, target_index)
        step_log_p = target.transition_log_prob(
            latent_history,
            latent,
            transition_condition,
            util.dynamic_index_pytree_in_dim(parameters_batched, step_index + 1, 0),
        )
        return None, step_log_p

    _, transition_log_ps = jax.lax.scan(
        scan_transition,
        None,
        xs=jnp.arange(transition_steps),
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
    sequence_length = _validate_sequence_lengths(
        target,
        x_path,
        observation_path,
        condition,
    )

    parameters_batched = util.broadcast_packable(
        parameters,
        leading_axis_len=sequence_length,
    )

    def scan_emission(_, step_index: jax.Array) -> tuple[None, Scalar]:
        latent_end_index = target.prior_order - 1 + step_index
        observation_index = target.observation_dependency + step_index

        latent_history = _latent_context_at(
            target,
            x_path,
            end_index=latent_end_index,
            order=target.emission_order,
        )
        observation = util.dynamic_index_pytree_in_dim(observation_path, observation_index, 0)
        observation_history = _observation_context_at(
            target,
            observation_path,
            end_index=observation_index - 1,
        )
        observation_condition = _condition_at(condition, observation_index)

        step_log_p = target.emission_log_prob(
            latent_history,
            observation,
            observation_history,
            observation_condition,
            util.dynamic_index_pytree_in_dim(parameters_batched, step_index, 0),
        )
        return None, step_log_p

    _, emission_log_ps = jax.lax.scan(
        scan_emission,
        None,
        xs=jnp.arange(sequence_length),
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
