"""Model evaluation utilities for computing log probabilities."""

import jax
from jaxtyping import Scalar
import typing

import seqjax.model.typing as seqjtyping
from seqjax.model.base import SequentialModel, Prior, Transition, Emission
from seqjax.util import (
    index_pytree,
    slice_pytree,
    broadcast_packable,
)


def slice_prior_conditions[
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    PriorConditionsT,
](
    condition: ConditionT,
    prior: Prior[typing.Any, PriorConditionsT, typing.Any],
) -> PriorConditionsT:
    """Slice the prior conditions type from the full condition type."""
    return typing.cast(
        PriorConditionsT,
        tuple(index_pytree(condition, ix) for ix in range(prior.order)),
    )


def slice_prior_latent[
    LatentT: seqjtyping.Latent,
    PriorLatentT,
](
    x_path: LatentT,
    prior: Prior[PriorLatentT, typing.Any, typing.Any],
) -> PriorLatentT:
    """Slice the prior conditions type from the full condition type."""
    return typing.cast(
        PriorLatentT,
        tuple(index_pytree(x_path, ix) for ix in range(prior.order)),
    )


def slice_transition_latent_history[
    LatentT: seqjtyping.Latent,
    TransitionLatentHistoryT,
](
    x_path: LatentT,
    transition: Transition[
        TransitionLatentHistoryT, typing.Any, typing.Any, typing.Any
    ],
    prior: Prior[typing.Any, typing.Any, typing.Any],
) -> TransitionLatentHistoryT:
    """
    Build transition latent history from prior latents.
    The first transition always targets x_1, so we define relative to x_0.
    For example, the emission may require x_-1, which we don't want to include if
    transition.order is 1.
    """
    sequence_start = prior.order - 1
    sequence_length = x_path.batch_shape[0] - sequence_start
    return typing.cast(
        TransitionLatentHistoryT,
        tuple(
            slice_pytree(
                x_path,
                sequence_start + 1 + i,
                sequence_start + sequence_length + i,
            )
            for i in range(-transition.order, 0)
        ),
    )


def slice_emission_latent_history[
    LatentT: seqjtyping.Latent,
    EmissionLatentHistoryT,
    ObservationHistoryT,
](
    x_path: LatentT,
    emission: Emission[
        EmissionLatentHistoryT,
        typing.Any,
        typing.Any,
        typing.Any,
        ObservationHistoryT,
    ],
    prior: Prior[typing.Any, typing.Any, typing.Any],
) -> EmissionLatentHistoryT:
    """
    Build transition latent history from prior latents.
    The first observation corresponds to x_0.
    """
    sequence_start = prior.order - 1
    sequence_length = x_path.batch_shape[0] - sequence_start
    return typing.cast(
        EmissionLatentHistoryT,
        tuple(
            slice_pytree(
                x_path,
                sequence_start + 1 + i,
                sequence_start + 1 + i + sequence_length,
            )
            for i in range(-emission.order, 0)
        ),
    )


def slice_emission_observation_history[
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT,
](
    observation_path: ObservationT,
    emission: Emission[
        typing.Any,
        typing.Any,
        typing.Any,
        typing.Any,
        ObservationHistoryT,
    ],
) -> ObservationHistoryT:
    """
    Build transition latent history from prior latents.
    The first observation corresponds to x_0.
    """
    sequence_start = emission.observation_dependency
    sequence_length = observation_path.batch_shape[0] - sequence_start
    return typing.cast(
        ObservationHistoryT,
        tuple(
            slice_pytree(
                observation_path, sequence_start + i, sequence_start + i + sequence_length
            )
            for i in range(-emission.observation_dependency, 0)
        ),
    )


def log_prob_x[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    PriorLatentT: tuple[seqjtyping.Latent, ...],
    PriorConditionT: tuple[seqjtyping.Condition, ...],
    TransitionLatentHistoryT: tuple[seqjtyping.Latent, ...],
    EmissionLatentHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
](
    target: SequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        PriorLatentT,
        PriorConditionT,
        TransitionLatentHistoryT,
        EmissionLatentHistoryT,
        ObservationHistoryT,
    ],
    x_path: LatentT,
    condition: ConditionT,
    parameters: ParametersT,
) -> Scalar:
    """Return ``log p(x)`` for a latent sequence.

    ``x_path`` should contain only the ``t \\geq 0`` portion of the latent
    sequence.  If ``target.prior.order > 1`` then the required history prior to
    ``t=0`` can be supplied via ``x_history``.

    Internally the function slices out the latent histories for each time step to
    allow a vectorised evaluation of the density.  This trades memory for speed
    and may require a more memory-efficient implementation for long sequences.
    """
    sequence_start = target.prior.order - 1
    x_shape = x_path.batch_shape
    sequence_length = x_shape[0] - sequence_start

    # batch parameters down sequence axis if needed
    parameters_batched = broadcast_packable(
        parameters,
        leading_axis_len=sequence_length,
    )

    # compute prior
    prior_latents = slice_prior_latent(x_path, target.prior)
    prior_conditions = slice_prior_conditions(condition, target.prior)
    log_p_x_0 = target.prior.log_prob(
        prior_latents, prior_conditions, index_pytree(parameters_batched, 0)
    )

    # rest of sequence
    # we need TransitionLatentHistoryT for t in [1, sequence_length - 1]
    # each element of the tuple is a lagged sequence of x
    latent_history = slice_transition_latent_history(
        x_path, target.transition, target.prior
    )
    target_latent = slice_pytree(
        x_path,
        sequence_start + 1,
        sequence_start + sequence_length,
    )
    transition_condition = slice_pytree(
        condition,
        sequence_start + 1,
        sequence_start + sequence_length,
    )

    transition_log_p_x = jax.vmap(target.transition.log_prob)(
        latent_history,
        target_latent,
        transition_condition,
        slice_pytree(
            parameters_batched,
            1,
            sequence_length,
        ),
    ).sum()

    return (log_p_x_0 + transition_log_p_x).sum()


def log_prob_y_given_x[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    PriorLatentT: tuple[seqjtyping.Latent, ...],
    PriorConditionT: tuple[seqjtyping.Condition, ...],
    TransitionLatentHistoryT: tuple[seqjtyping.Latent, ...],
    EmissionLatentHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
](
    target: SequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        PriorLatentT,
        PriorConditionT,
        TransitionLatentHistoryT,
        EmissionLatentHistoryT,
        ObservationHistoryT,
    ],
    x_path: LatentT,
    observation_path: ObservationT,
    condition: ConditionT,
    parameters: ParametersT,
) -> Scalar:
    """Return ``log p(y | x)`` for a sequence of observations.

    ``x_path`` and ``observation_path`` share the same leading ``Batch`` dimensions,
    matching the output of :func:`~seqjax.model.simulate.simulate`.

    The prior order defines x_0 and the transition observation dependency defines
    y_0.
    """
    x_length = x_path.batch_shape[0]

    x_sequence_start = target.prior.order - 1
    y_sequence_start = target.emission.observation_dependency

    # this is the length of the observation sequence
    # should == y_sequence_length - y_sequence_start
    sequence_length = x_length - x_sequence_start

    # batch parameters down sequence axis if needed
    parameters_batched = broadcast_packable(
        parameters,
        leading_axis_len=sequence_length,
    )

    latent_history = slice_emission_latent_history(
        x_path, target.emission, target.prior
    )

    emission_history = slice_emission_observation_history(
        observation_path,
        target.emission,
    )

    observations = slice_pytree(
        observation_path, y_sequence_start, sequence_length + y_sequence_start
    )
    observation_conditions = slice_pytree(
        condition, y_sequence_start, sequence_length + y_sequence_start
    )

    return jax.vmap(
        target.emission.log_prob,
    )(
        latent_history,
        observations,
        emission_history,
        observation_conditions,
        parameters_batched,
    ).sum()


def log_prob_joint[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    PriorLatentT: tuple[seqjtyping.Latent, ...],
    PriorConditionT: tuple[seqjtyping.Condition, ...],
    TransitionLatentHistoryT: tuple[seqjtyping.Latent, ...],
    EmissionLatentHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
](
    target: SequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        PriorLatentT,
        PriorConditionT,
        TransitionLatentHistoryT,
        EmissionLatentHistoryT,
        ObservationHistoryT,
    ],
    x_path: LatentT,
    observation_path: ObservationT,
    condition: ConditionT,
    parameters: ParametersT,
) -> Scalar:
    """Return ``log p(x, y)`` for a path and observations.

    ``x_path`` should contain only the ``t \\geq 0`` portion of the latent path.
    Pass ``x_history`` to supply any earlier latent values required by
    ``target.prior`` or the emission model.

    The latent and observation sequences share their ``Batch`` dimensions,
    reflecting the output of :func:`~seqjax.model.simulate.simulate`.
    """
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
