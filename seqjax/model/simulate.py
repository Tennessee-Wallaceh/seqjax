"""Utilities for simulating sequences from a target model."""

import typing

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

from seqjax.model.base import SequentialModel, Prior
import seqjax.model.typing as seqjtyping
from seqjax.util import concat_pytree, index_pytree, slice_pytree


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


def step[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
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
    parameters: ParametersT,
    state: tuple[
        PriorLatentT,
        ObservationHistoryT,
    ],
    inputs,
) -> tuple[
    tuple[
        PriorLatentT,
        ObservationHistoryT,
    ],
    tuple[LatentT, ObservationT],
]:
    """Single simulation step returning updated state and new sample."""

    # pad the RHS with a None, then no condition => condition is None
    step_key, condition = (inputs + (None,))[:2]
    latents, emissions = state
    transition_key, emission_key = jrandom.split(step_key)
    latent_history = target.latent_view_for_transition(latents)

    # last latent is at t
    # sample x_t+1 then y_t+1
    next_latent = target.transition.sample(
        transition_key,
        latent_history,
        condition,
        parameters,
    )

    latents = target.add_latent_history(latents, next_latent)
    emission_p_history = target.latent_view_for_emission(latents)
    emission = target.emission.sample(
        emission_key,
        emission_p_history,
        emissions,
        condition,
        parameters,
    )

    emission_history = target.add_observation_history(emissions, emission)

    return (latents, emission_history), (next_latent, emission)


def simulate[
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
    key: PRNGKeyArray,
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
    parameters: ParametersT,
    sequence_length: int,
    condition: ConditionT | None = None,
    reference_emission: ObservationHistoryT | None = None,
) -> tuple[
    LatentT,
    ObservationT,
]:
    """Simulate a path of length ``sequence_length`` from ``target``.

    The returned latent and observation arrays contain only ``x_0 -> x_{T-1}``
    and ``y_0 -> y_{T-1}`` for ``T = sequence_length``.  Any additional
    history required by the transition or emission is handled internally:

    * The :class:`~seqjax.model.base.Prior` supplies the latent states for
      ``t < 0``.  The number of states is ``target.prior.order`` and depends on
      both the transition and emission orders.
    * Initial observations ``y_{t<0}`` are provided via
      ``reference_emission``.

    For example:

    #. ``obs_dep=1`` (emission depends on the previous observation) requires one
       ``y_{-1}`` but only ``x_0`` from the prior.
    #. ``latent_dep=1`` with a first-order transition needs ``x_{-1}`` and
       ``x_0`` from the prior.
    #. ``latent_dep=1`` with a second-order transition requires
       ``x_{-2}, x_{-1}, x_0``.

    The ``condition`` array must therefore have length
    ``sequence_length + target.prior.order - 1`` so that the prior and every
    subsequent transition step receive the correct context.

    Returns ``(latents, observations)``.
    ``latents`` and ``observations`` share the same leading ``Batch`` dimensions
    while ``SequenceAxis`` corresponds to ``sequence_length``.

    ``latents`` will have length ``sequence_length + target.prior.order - 1`` along
    the ``SequenceAxis`` to account for the prior states.  ``observations`` will have length
    ``sequence_length``, as no additional observations are needed beyond those.
    """

    if sequence_length < 1:
        raise jax.errors.JaxRuntimeError(
            f"sequence_length must be >= 1, got {sequence_length}"
        )

    if condition is None:
        if target.condition_cls is seqjtyping.NoCondition:
            condition = typing.cast(ConditionT, seqjtyping.NoCondition())
        else:
            raise jax.errors.JaxRuntimeError(
                "condition cannot be None for models with a condition"
            )

    if target.condition_cls is not seqjtyping.NoCondition:
        cond_length = condition.batch_shape[0]
        required_length = sequence_length + target.prior.order - 1
        if cond_length < required_length:
            raise jax.errors.JaxRuntimeError(
                "condition must have length >= {} for sequence_length {}".format(
                    required_length,
                    sequence_length,
                )
            )
    if reference_emission is None:
        reference_emission = typing.cast(ObservationHistoryT, ())
    elif len(reference_emission) != target.emission.observation_dependency:
        raise jax.errors.JaxRuntimeError(
            "Reference emission must match emission.observation_dependency"
        )

    init_x_key, init_y_key, *step_keys = jrandom.split(key, sequence_length + 1)

    # special handling for sampling first state
    prior_conditions = slice_prior_conditions(condition, target.prior)

    condition_0 = index_pytree(condition, target.prior.order - 1)

    # The prior will produce the maximal latent history
    x_0 = target.prior.sample(init_x_key, prior_conditions, parameters)
    y_0 = target.emission.sample(
        init_y_key,
        target.latent_view_for_emission(x_0),
        reference_emission,
        condition_0,
        parameters,
    )

    # build start point of scan
    emission_history = target.add_observation_history(reference_emission, y_0)
    state = (x_0, emission_history)
    inputs = (
        (jnp.array(step_keys),)
        if condition is None
        else (
            jnp.array(step_keys),
            slice_pytree(
                condition,
                target.prior.order,
                sequence_length + target.prior.order - 1,
            ),
        )
    )

    # scan for generic step
    _, (latent_path, observed_path) = jax.lax.scan(
        partial(step, target, parameters),
        state,
        xs=inputs,
        length=sequence_length - 1,
    )

    # add the starting values back
    latent_full = concat_pytree(
        *typing.cast(
            tuple[LatentT, ...], x_0
        ),  # we know x_0 is just a tuple of LatentT, we just can't express that in typing
        latent_path,
    )
    observed_full = concat_pytree(
        y_0,
        observed_path,
    )

    return latent_full, observed_full
