"""Utilities for simulating sequences from a target model."""


import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

from seqjax.model import (
    interface as model_interface,
    util as model_util
)
from seqjax import util
import seqjax.model.typing as seqjtyping
from seqjax.model.condition import layout_for, normalize_condition_path

def step[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    parameters: ParametersT,
    state: tuple[
        model_interface.LatentContext[LatentT],
        model_interface.ObservationContext[ObservationT],
    ],
    inputs: tuple[PRNGKeyArray, ConditionT, ConditionT],
) -> tuple[
    tuple[
        model_interface.LatentContext[LatentT],
        model_interface.ObservationContext[ObservationT],
    ],
    tuple[LatentT, ObservationT],
]:
    """Single simulation step returning updated state and new sample."""

    step_key, transition_condition, emission_condition = inputs
    latents, observation_history = state
    transition_key, emission_key = jrandom.split(step_key)

    # last latent is at t
    # sample x_t+1 then y_t+1
    next_latent = target.transition_sample(
        transition_key,
        latents,
        transition_condition,
        parameters,
    )

    latents = model_util.add_history(latents, next_latent)
    emission = target.emission_sample(
        emission_key,
        latents,
        observation_history,
        emission_condition,
        parameters,
    )
    observation_history = model_util.add_history(observation_history, emission)

    return (latents, observation_history), (next_latent, emission)

def simulate[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    key: PRNGKeyArray,
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    parameters: ParametersT,
    sequence_length: int,
    condition: ConditionT | None = None,
    observation_history: model_interface.ObservationContext[ObservationT] =  model_interface.ObservationContext.from_values(length=0)
):
    if sequence_length < 1:
        raise jax.errors.JaxRuntimeError(
            f"sequence_length must be >= 1, got {sequence_length}"
        )
    
    condition = normalize_condition_path(target, condition, (sequence_length,))

    init_x_key, init_y_key, *step_keys = jrandom.split(key, sequence_length + 1)

    condition_layout = layout_for(target)
    prepared_conditions = condition_layout.prepare(target, condition, sequence_length)
    latent_context = target.prior_sample(
        init_x_key, prepared_conditions.prior, parameters
    )

    initial_obs = target.emission_sample(
        init_y_key,
        latent_context,
        observation_history,
        prepared_conditions.initial_emission,
        parameters
    )

    observation_history = model_util.add_history(observation_history, initial_obs)

    init_state = (latent_context, observation_history)

    inputs = (
        jnp.array(step_keys),
        prepared_conditions.transitions,
        prepared_conditions.recurrent_emissions,
    )

    def model_step(
        state: tuple[
            model_interface.LatentContext[LatentT],
            model_interface.ObservationContext[ObservationT],
        ],
        inputs: tuple[PRNGKeyArray, ConditionT, ConditionT],
    ) -> tuple[
        tuple[
            model_interface.LatentContext[LatentT],
            model_interface.ObservationContext[ObservationT],
        ],
        tuple[LatentT, ObservationT],
    ]:
        return step(target, parameters, state, inputs)

    (_, _), (latent_scan, obs_scan) = jax.lax.scan(
        model_step,
        init_state,
        xs=inputs,
        length=sequence_length - 1,
        unroll=1
    )

    latent_full = util.concat_pytree(
        *latent_context.values,
        latent_scan,
    )
    observed_full = util.concat_pytree(
        initial_obs,
        obs_scan,
    )
    return latent_full, observed_full
