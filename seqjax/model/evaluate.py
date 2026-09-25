"""Model evaluation utilities for computing log probabilities."""

import typing

import jax
import jax.numpy as jnp
from jaxtyping import Scalar

import seqjax.model.typing as seqjtyping
from seqjax import util
from seqjax.model import interface as model_interface
from seqjax.model import util as model_util

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
    x_prior: LatentT,
    x_path: LatentT,
    parameters: ParametersT,
    condition: ConditionT | None = None,
    observation_history: model_interface.ObservedHistoryContext[
        ObservationT,
        ConditionT,
    ] | None = None,
    # Supports past-observation -> latent dependence.
    observation_path: ObservationT | None = None,
) -> Scalar:
    """Return ``log p(x_path | x_prior)``."""

    sequence_length = x_path.batch_shape[0]

    condition = model_util.normalize_condition_path(
        target,
        condition,
        (sequence_length,),
    )

    if condition.batch_shape != (sequence_length,):
        raise ValueError(
            "condition and x_path have different batch shapes: "
            f"{condition.batch_shape=} {x_path.batch_shape=}"
        )

    if observation_history is None:
        if target.observation_context_length != 0:
            raise ValueError(
                "observation_history must be provided when "
                f"target.observation_context_length="
                f"{target.observation_context_length}"
            )

        observation_history = target.observed_history_context()

    elif observation_history.length != target.observation_context_length:
        raise ValueError(
            "observation_history has the wrong length: "
            f"expected {target.observation_context_length}, "
            f"received {observation_history.length}"
        )

    required_observation_length = max(sequence_length - 1, 0)

    if observation_path is not None:
        if len(observation_path.batch_shape) != 1:
            raise ValueError(
                "Expected a single observation sequence, received "
                f"{observation_path.batch_shape=}"
            )

        observation_length = observation_path.batch_shape[0]

        if (
            target.transition_observation_order > 0
            and observation_length < required_observation_length
        ):
            raise ValueError(
                "observation_path is too short for latent evaluation: "
                f"required at least {required_observation_length}, "
                f"received {observation_length}"
            )

    elif (
        target.transition_observation_order > 0
        and required_observation_length > 0
    ):
        raise ValueError(
            "observation_path must contain the observations preceding "
            "the evaluated transitions"
        )

    transition_history = model_util.batch_latent_context(
        target,
        x_prior,
        x_path,
        context_end="last",
    )

    latent_history_in_axes = (
        0 if target.latent_context_length > 0 else None
    )

    if target.transition_observation_order > 0:
        #TODO: Use model_util batching
        history_length = target.observation_context_length

        observed_prior = jax.tree.map(
            lambda *values: jnp.stack(values, axis=0),
            *observation_history.values,
        )

        if required_observation_length > 0:
            assert observation_path is not None

            preceding_observations = util.slice_pytree(
                observation_path,
                0,
                required_observation_length,
            )
            preceding_conditions = util.slice_pytree(
                condition,
                0,
                required_observation_length,
            )

            observed_path = model_interface.ObservedItem(
                observation=preceding_observations,
                condition=preceding_conditions,
            )

            observed_full = jax.tree.map(
                lambda prior, path: jnp.concatenate(
                    (prior, path),
                    axis=0,
                ),
                observed_prior,
                observed_path,
            )
        else:
            observed_full = observed_prior

        transition_observation_history = (
            target.observed_history_context(
                *(
                    util.slice_pytree(
                        observed_full,
                        history_length + lag,
                        history_length + lag + sequence_length,
                    )
                    for lag in range(-history_length, 0)
                )
            )
        )
        observation_history_in_axes = 0

    else:
        # The transition declares no observation-history dependence, so the
        # unchanged context can be shared across all mapped evaluations.
        transition_observation_history = observation_history
        observation_history_in_axes = None

    transition_log_ps = jax.vmap(
        target.transition_log_prob,
        in_axes=(
            latent_history_in_axes,
            0,
            None,
            0,
            observation_history_in_axes,
        ),
    )(
        transition_history,
        x_path,
        parameters,
        condition,
        transition_observation_history,
    )

    return transition_log_ps.sum()

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
    x_prior: LatentT,
    x_path: LatentT,
    observation_path: ObservationT,
    parameters: ParametersT,
    condition: ConditionT | None = None,
    observation_history: model_interface.ObservedHistoryContext[
        ObservationT,
        ConditionT,
    ] | None = None,
) -> Scalar:
    """Return ``log p(observation_path | x_prior, x_path)``."""

    sequence_length = x_path.batch_shape[0]
    condition = model_util.normalize_condition_path(
        target,
        condition,
        (sequence_length,),
    )

    if observation_path.batch_shape != (sequence_length,):
        raise ValueError(
            "x_path and observation_path have different batch shapes: "
            f"{x_path.batch_shape=} "
            f"{observation_path.batch_shape=}"
        )

    if condition.batch_shape != (sequence_length,):
        raise ValueError(
            "condition and x_path have different batch shapes: "
            f"{condition.batch_shape=} {x_path.batch_shape=}"
        )

    if observation_history is None:
        if target.observation_context_length != 0:
            raise ValueError(
                "observation_history must be provided when "
                f"target.observation_context_length="
                f"{target.observation_context_length}"
            )

        observation_history = target.observed_history_context()

    elif observation_history.length != target.observation_context_length:
        raise ValueError(
            "observation_history has the wrong length: "
            f"expected {target.observation_context_length}, "
            f"received {observation_history.length}"
        )

    emission_latent_history = model_util.batch_latent_context(
        target,
        x_prior,
        x_path,
        context_end="current",
    )

    emission_observation_history = model_util.batch_observation_history(
        target,
        observation_history,
        observation_path,
        condition,
    )

    latent_history_in_axes = (
        0 if target.latent_context_length > 0 else None
    )
    observation_history_in_axes = (
        0 if target.observation_context_length > 0 else None
    )

    emission_log_ps = jax.vmap(
        target.emission_log_prob,
        in_axes=(
            latent_history_in_axes,
            0,
            None,
            0,
            observation_history_in_axes,
        ),
    )(
        emission_latent_history,
        observation_path,
        parameters,
        condition,
        emission_observation_history,
    )

    return emission_log_ps.sum()

def log_prob_x_prior[
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
    x_prior: LatentT,
    parameters: ParametersT,
) -> Scalar:
    """Return ``log p(x_prior)``."""
    x_prior_context = target.latent_context(
        *(
            util.index_pytree(x_prior, index)
            for index in range(target.prior_latent_order)
        )
    )

    if x_prior.length != target.prior_latent_order:
        raise ValueError(
            "x_prior has the wrong length: "
            f"expected {target.prior_latent_order}, "
            f"received {x_prior.length}"
        )

    return target.prior_log_prob(x_prior_context, parameters)

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
    x_prior: LatentT,
    x_path: LatentT,
    observation_path: ObservationT,
    parameters: ParametersT,
    condition: ConditionT | None = None,
    observation_history: model_interface.ObservedHistoryContext[
        ObservationT,
        ConditionT,
    ] | None = None,
) -> Scalar:
    """Return ``log p(x_prior, x_path, observation_path)``."""


    return (
        log_prob_x_prior(
            target,
            x_prior,
            parameters,
        )
        + log_prob_x(
            target,
            x_prior,
            x_path,
            parameters,
            condition=condition,
            observation_history=observation_history,
            observation_path=observation_path,
        )
        + log_prob_y_given_x(
            target,
            x_prior,
            x_path,
            observation_path,
            parameters,
            condition=condition,
            observation_history=observation_history,
        )
    )