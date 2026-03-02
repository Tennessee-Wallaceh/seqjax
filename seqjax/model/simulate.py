"""Utilities for simulating sequences from method-based target models."""

import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import seqjax.model.typing as seqjtyping
from seqjax.model.base import SequentialModelBase
from seqjax.util import concat_pytree, index_pytree, slice_pytree


def simulate[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    key: PRNGKeyArray,
    target: SequentialModelBase[LatentT, ObservationT, ConditionT, ParametersT],
    parameters: ParametersT,
    sequence_length: int,
    condition: ConditionT | None = None,
    reference_emission: tuple[ObservationT, ...] | None = None,
) -> tuple[LatentT, ObservationT]:
    """Simulate using the method-based model API."""

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
        required_length = sequence_length + target.prior_order - 1
        if cond_length < required_length:
            raise jax.errors.JaxRuntimeError(
                "condition must have length >= {} for sequence_length {}".format(
                    required_length,
                    sequence_length,
                )
            )

    if reference_emission is None:
        reference_emission = ()
    elif len(reference_emission) != target.observation_dependency:
        raise jax.errors.JaxRuntimeError(
            "Reference emission must match observation_dependency"
        )

    init_x_key, init_y_key, *step_keys = jrandom.split(key, sequence_length + 1)

    prior_conditions = target.make_condition_context(
        *tuple(index_pytree(condition, ix) for ix in range(target.prior_order))
    )
    condition_0 = index_pytree(condition, target.prior_order - 1)

    latent_context = target.prior_sample(init_x_key, prior_conditions, parameters)
    latent_context = target.make_latent_context(*latent_context.to_tuple())
    if latent_context.current_length != target.prior_order:
        raise jax.errors.JaxRuntimeError(
            f"prior_sample must produce latent history length {target.prior_order}, got {latent_context.current_length}"
        )

    y_0 = target.emission_sample(
        init_y_key,
        latent_context,
        target.make_observation_context(*reference_emission),
        condition_0,
        parameters,
    )
    observation_context = target.make_observation_context(*reference_emission).append(y_0)

    latents = latent_context.to_tuple()
    if sequence_length > 1:
        scan_conditions = slice_pytree(
            condition,
            target.prior_order,
            target.prior_order + sequence_length - 1,
        )

        def _scan_step(
            state: tuple[tuple[LatentT, ...], tuple[ObservationT, ...]],
            inputs: tuple[PRNGKeyArray, ConditionT],
        ) -> tuple[
            tuple[tuple[LatentT, ...], tuple[ObservationT, ...]],
            tuple[LatentT, ObservationT],
        ]:
            latent_values, observation_values = state
            step_key, step_condition = inputs
            transition_key, emission_key = jrandom.split(step_key)

            latent_context_step = target.make_latent_context(*latent_values)
            observation_context_step = target.make_observation_context(*observation_values)

            next_latent = target.transition_sample(
                transition_key,
                latent_context_step,
                step_condition,
                parameters,
            )
            latent_context_step = latent_context_step.append(next_latent)
            y_next = target.emission_sample(
                emission_key,
                latent_context_step,
                observation_context_step,
                step_condition,
                parameters,
            )
            observation_context_step = observation_context_step.append(y_next)

            return (
                latent_context_step.to_tuple(),
                observation_context_step.to_tuple(),
            ), (next_latent, y_next)

        init_state = (latent_context.to_tuple(), observation_context.to_tuple())
        (_, _), (latent_scan, obs_scan) = jax.lax.scan(
            _scan_step,
            init_state,
            xs=(jnp.array(step_keys), scan_conditions),
            length=sequence_length - 1,
        )
        latent_full = concat_pytree(*latents, latent_scan)
        observed_full = concat_pytree(y_0, obs_scan)
    else:
        latent_full = concat_pytree(*latents)
        observed_full = y_0

    return typing.cast(LatentT, latent_full), typing.cast(ObservationT, observed_full)


simulate_methods = simulate
