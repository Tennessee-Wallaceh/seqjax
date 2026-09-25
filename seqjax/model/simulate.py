"""Utilities for simulating sequences from a target model."""


import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

from seqjax.model import interface as model_interface
from seqjax.model import util as model_util
import seqjax.model.typing as seqjtyping

def step[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ],
    parameters: ParametersT,
    state: tuple[
        model_interface.LatentContext[LatentT, LatentContextLength],
        model_interface.ObservedHistoryContext[
            ObservationT, 
            ConditionT,
            ObservationContextLength,
        ],
    ],
    inputs: tuple[PRNGKeyArray, ConditionT],
) -> tuple[
    tuple[
        model_interface.LatentContext[LatentT, LatentContextLength],
        model_interface.ObservedHistoryContext[
            ObservationT, 
            ConditionT,
            ObservationContextLength,
        ],
    ],
    tuple[LatentT, ObservationT],
]:
    """Single simulation step returning updated state and new sample."""

    step_key, condition = inputs
    latents, observation_history = state
    transition_key, emission_key = jrandom.split(step_key)

    # last latent is at t
    # sample x_t+1 then y_t+1
    next_latent = target.transition_sample(
        transition_key,
        latents,
        parameters,
        condition,
        observation_history,
    )

    observation = target.emission_sample(
        emission_key,
        next_latent,
        parameters,
        condition,
        latents,
        observation_history,
    )
    observation_history = observation_history.append_observation(
        observation,
        condition,
    )
    latents = latents.append(next_latent)

    return (latents, observation_history), (next_latent, observation)

def simulate[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
](
    key: PRNGKeyArray,
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ],
    parameters: ParametersT,
    *,
    sequence_length: int | None = None,
    condition: ConditionT | None = None,
    observation_history: model_interface.ObservedHistoryContext[
        ObservationT, ConditionT, ObservationContextLength
    ] =  None
) -> tuple[
    model_interface.LatentContext[LatentT, LatentContextLength],
    LatentT,
    ObservationT,
]:
    if (sequence_length is None) == (condition is None):
        raise ValueError("Exactly one of sequence_length and condition must be provided") 
    elif condition is None:
        condition = model_util.normalize_condition_path(target, condition, (sequence_length,))
    elif sequence_length is None:
        if len(condition.batch_shape) != 1:
            raise ValueError(
                "Simulation is defined for single sequences, received "
                f"condition.batch_shape={condition.batch_shape}"
            )
        sequence_length = condition.batch_shape[0]

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

    init_x_key, *step_keys = jrandom.split(key, sequence_length + 1)

    prior_context = target.prior_sample(init_x_key, parameters)

    init_state = (prior_context, observation_history)

    inputs = (jnp.array(step_keys), condition)

    # closes over target and parameters
    def model_step(
        state: tuple[
            model_interface.LatentContext[LatentT],
            model_interface.ObservedHistoryContext[ObservationT, ConditionT],
        ],
        inputs: tuple[PRNGKeyArray, ConditionT],
    ) -> tuple[
        tuple[
            model_interface.LatentContext[LatentT],
            model_interface.ObservedHistoryContext[ObservationT, ConditionT],
        ],
        tuple[LatentT, ObservationT],
    ]:
        return step(target, parameters, state, inputs)

    (_, _), (latent_scan, obs_scan) = jax.lax.scan(
        model_step,
        init_state,
        xs=inputs,
        length=sequence_length,
        unroll=1
    )

    return prior_context, latent_scan, obs_scan
