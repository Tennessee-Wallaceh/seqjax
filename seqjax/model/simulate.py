"""Utilities for simulating sequences from a target model."""

import typing

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray, PyTree

from seqjax.model.base import (
    SequentialModel,
)
import seqjax.model.typing as seqjtyping
from seqjax.util import concat_pytree, index_pytree, slice_pytree


def step[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...] | None,
    ConditionT: seqjtyping.Condition | None,
    ParametersT: seqjtyping.Parameters,
](
    target: SequentialModel[
        ParticleT,
        InitialParticleT,
        TransitionParticleHistoryT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
        ConditionT,
        ParametersT,
    ],
    parameters: ParametersT,
    state: tuple[
        tuple[ParticleT, ...],
        ObservationHistoryT,
    ],
    inputs,
) -> tuple[
    tuple[tuple[ParticleT, ...], ObservationHistoryT],
    tuple[ParticleT, ObservationT],
]:
    """Single simulation step returning updated state and new sample."""

    # pad the RHS with a None, then no condition => condition is None
    step_key, condition = (inputs + (None,))[:2]
    particles, emissions = state
    transition_key, emission_key = jrandom.split(step_key)
    particle_history = typing.cast(
        TransitionParticleHistoryT, particles[-target.transition.order :]
    )

    # last particle is at t
    # sample x_t+1 then y_t+1
    next_particle = target.transition.sample(
        transition_key,
        particle_history,
        condition,
        parameters,
    )

    particles = (*particles, next_particle)
    emission_p_history = typing.cast(
        ObservationParticleHistoryT, particles[-target.emission.order :]
    )
    emission = target.emission.sample(
        emission_key,
        emission_p_history,
        emissions,
        condition,
        parameters,
    )

    # add the next particle to the history before emission
    # only pass on necessary information
    # # read off histories of appropriate order
    max_latent_order = max(target.transition.order, target.emission.order)
    particle_history = particles[-max_latent_order:]
    emission_history = (*emissions, emission)
    emission_history = typing.cast(
        ObservationHistoryT,
        emission_history[
            len(emission_history) - target.emission.observation_dependency :
        ],
    )

    return (particle_history, emission_history), (next_particle, emission)


def simulate[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...] | None,
    ConditionT: seqjtyping.Condition | None,
    ParametersT: seqjtyping.Parameters,
](
    key: PRNGKeyArray,
    target: SequentialModel[
        ParticleT,
        InitialParticleT,
        TransitionParticleHistoryT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
        ConditionT,
        ParametersT,
    ],
    condition: PyTree | None,
    parameters: ParametersT,
    sequence_length: int,
) -> tuple[
    ParticleT,
    ObservationT,
    InitialParticleT,
    ObservationHistoryT,
]:
    """Simulate a path of length ``sequence_length`` from ``target``.

    The returned latent and observation arrays contain only ``x_0 -> x_{T-1}``
    and ``y_0 -> y_{T-1}`` for ``T = sequence_length``.  Any additional
    history required by the transition or emission is handled internally:

    * The :class:`~seqjax.model.base.Prior` supplies the latent states for
      ``t < 0``.  The number of states is ``target.prior.order`` and depends on
      both the transition and emission orders.
    * Initial observations ``y_{t<0}`` are provided via
      ``parameters.reference_emission``.

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

    Returns ``(latents, observations, latent_history, observation_history)``.
    ``latents`` and ``observations`` share the same leading ``Batch`` dimensions
    while ``SequenceAxis`` corresponds to ``sequence_length``.  The final two
    values are the latent and observation histories used for the simulation.
    """

    if sequence_length < 1:
        raise jax.errors.JaxRuntimeError(
            f"sequence_length must be >= 1, got {sequence_length}"
        )

    if condition is not None:
        cond_length = condition.batch_shape[0]
        required_length = sequence_length + target.prior.order - 1
        if cond_length < required_length:
            raise jax.errors.JaxRuntimeError(
                "condition must have length >= {} for sequence_length {}".format(
                    required_length,
                    sequence_length,
                )
            )
    init_x_key, init_y_key, *step_keys = jrandom.split(key, sequence_length + 1)

    # special handling for sampling first state
    prior_conditions = tuple(
        index_pytree(condition, ix) for ix in range(target.prior.order)
    )
    condition_0 = index_pytree(condition, target.prior.order - 1)
    x_0 = target.prior.sample(init_x_key, prior_conditions, parameters)
    y_0 = target.emission.sample(
        init_y_key,
        x_0,
        parameters.reference_emission,
        condition_0,
        parameters,
    )

    # build start point of scan
    emission_history = (*parameters.reference_emission, y_0)  # type: ignore[assignment]
    emission_history = tuple(  # type: ignore[assignment]
        emission_history[
            len(emission_history) - target.emission.observation_dependency :
        ]
    )
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
        partial(step, target, parameters),  # type: ignore[arg-type]
        state,
        xs=inputs,
        length=sequence_length - 1,
    )

    latent_full = concat_pytree(*x_0, latent_path)
    observed_full = concat_pytree(
        *parameters.reference_emission,
        y_0,
        observed_path,
    )

    latent_start = target.prior.order - 1
    obs_start = target.emission.observation_dependency
    latent_path = slice_pytree(
        latent_full, latent_start, latent_start + sequence_length
    )
    observed_path = slice_pytree(
        observed_full,
        obs_start,
        obs_start + sequence_length,
    )

    latent_history = slice_pytree(latent_full, 0, latent_start)
    observation_history = slice_pytree(observed_full, 0, obs_start)

    return latent_path, observed_path, latent_history, observation_history
