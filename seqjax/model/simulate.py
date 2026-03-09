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
    inputs: tuple[PRNGKeyArray, ConditionT],
) -> tuple[
    tuple[
        model_interface.LatentContext[LatentT],
        model_interface.ObservationContext[ObservationT],
    ],
    tuple[LatentT, ObservationT],
]:
    """Single simulation step returning updated state and new sample."""

    # pad the RHS with a None, then no condition => condition is None
    step_key, condition = inputs
    latents, observation_history = state
    transition_key, emission_key = jrandom.split(step_key)

    # last latent is at t
    # sample x_t+1 then y_t+1
    next_latent = target.transition_sample(
        transition_key,
        latents,
        condition,
        parameters,
    )

    latents = model_util.add_history(latents, next_latent)
    emission = target.emission_sample(
        emission_key,
        latents,
        observation_history,
        condition,
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
    condition: ConditionT,
    observation_history: model_interface.ObservationContext[ObservationT] =  model_interface.ObservationContext.from_values(length=0)
):
    if sequence_length < 1:
        raise jax.errors.JaxRuntimeError(
            f"sequence_length must be >= 1, got {sequence_length}"
        )
    
    init_x_key, init_y_key, *step_keys = jrandom.split(key, sequence_length + 1)

    prior_conditions = model_util.slice_prior_context(
        target,
        condition,
    )
    initial_condition = model_util.initial_context(target, condition)
    latent_context = target.prior_sample(init_x_key, prior_conditions, parameters)

    initial_obs = target.emission_sample(
        init_y_key,
        latent_context,
        observation_history,
        initial_condition,
        parameters
    )

    observation_history = model_util.add_history(observation_history, initial_obs)

    init_state = (latent_context, observation_history)

    inputs = (
        (jnp.array(step_keys), (seqjtyping.NoCondition(),)* (sequence_length - 1))
        if isinstance(condition, seqjtyping.NoCondition)
        else (
            jnp.array(step_keys),
            util.slice_pytree(
                condition,
                target.prior_order,
                sequence_length + target.prior_order - 1,
            ),
        )
    )

    def model_step(
        state: tuple[
            model_interface.LatentContext[LatentT],
            model_interface.ObservationContext[ObservationT],
        ],
        inputs: tuple[PRNGKeyArray, ConditionT],
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

    return latent_scan, obs_scan
