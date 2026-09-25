import typing

import jax

import seqjax.model.typing as seqjtyping
from seqjax.model import (
    interface as model_interface,
)
from seqjax import util

def batch_latent_context[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        LatentContextLength,
        typing.Any,
    ],
    x_prior: LatentT,
    x_path: LatentT,
    *,
    context_end: typing.Literal["previous", "current"],
) -> model_interface.LatentContext[LatentT, LatentContextLength]:
    
    match context_end:
        case "previous":
            end_shift = 0
        case "current":
            end_shift = 1
        case _:
            raise ValueError(
                f"Invalid context_end={context_end!r}"
            )

    prior_length = x_prior.batch_shape[0]
    sequence_length = x_path.batch_shape[0]
    context_length = target.latent_context_length

    if prior_length != target.prior_latent_order:
        raise ValueError(
            "x_prior has the wrong length: "
            f"expected {target.prior_latent_order}, "
            f"received {prior_length}"
        )

    if context_length == 0:
        return target.latent_context()

    x_full = jax.tree.map(
        lambda x_prior_leaf, x_path_leaf: jnp.concatenate(
            (x_prior_leaf, x_path_leaf),
            axis=0,
        ),
        x_prior,
        x_path,
    )

    return target.latent_context(
        *(
            util.slice_pytree(
                x_full,
                prior_length + lag + end_shift,
                prior_length + lag + end_shift + sequence_length,
            )
            for lag in range(-context_length, 0)
        )
    )

def batch_observation_history[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    ObservationContextLength: int,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        typing.Any,
        ObservationContextLength,
    ],
    observation_history: model_interface.ObservedHistoryContext[
        ObservationT,
        ConditionT,
        ObservationContextLength,
    ],
    observation_path: ObservationT,
    condition: ConditionT,
) -> model_interface.ObservedHistoryContext[
    ObservationT,
    ConditionT,
    ObservationContextLength,
]:
    history_length = target.observation_context_length
    sequence_length = observation_path.batch_shape[0]

    if observation_history.length != history_length:
        raise ValueError(
            "observation_history has the wrong length: "
            f"expected {history_length}, "
            f"received {observation_history.length}"
        )

    if history_length == 0:
        return target.observation_context()

    observed_prior = jax.tree.map(
        lambda *values: jnp.stack(values, axis=0),
        *observation_history.values,
    )

    observed_path = model_interface.ObservedItem(
        observation=observation_path,
        condition=condition,
    )

    observed_full = jax.tree.map(
        lambda prior, path: jnp.concatenate(
            (prior, path),
            axis=0,
        ),
        observed_prior,
        observed_path,
    )

    return target.observed_history_context(
        *(
            util.slice_pytree(
                observed_full,
                history_length + lag,
                history_length + lag + sequence_length,
            )
            for lag in range(-history_length, 0)
        )
    )

def normalize_observation_history[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    ObservationContextLength: int,
](
    target: model_interface.SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        typing.Any,
    ],
    observation_history: (
        model_interface.ObservedHistoryContext[
            ObservationT,
            ConditionT,
            ObservationContextLength,
        ]
        | None
    ),
) -> model_interface.ObservedHistoryContext[
    ObservationT,
    ConditionT,
    ObservationContextLength,
]:
    expected_length = target.observation_context_length

    if observation_history is None:
        if expected_length != 0:
            raise ValueError(
                "observation_history is required because "
                f"target.observation_context_length={expected_length}"
            )

        return target.observed_history_context()

    if observation_history.length != expected_length:
        raise ValueError(
            "observation_history has the wrong length: "
            f"expected {expected_length}, "
            f"received {observation_history.length}"
        )

    return observation_history

def normalize_condition_path[ConditionT: seqjtyping.Condition](
    model: model_interface.SequentialModelProtocol[
        typing.Any,
        typing.Any,
        ConditionT,
        typing.Any,
    ],
    path: ConditionT | None,
    batch_shape: tuple[int, ...],
) -> ConditionT:
    """Materialize an omitted condition path at a model execution boundary."""
    condition_cls = model.condition_cls
    if path is None:
        if condition_cls is not seqjtyping.NoCondition:
            model_name = getattr(model, "__name__", type(model).__name__)
            raise ValueError(
                f"{model_name} requires a condition path, but condition=None was supplied."
            )
        return typing.cast(
            ConditionT,
            seqjtyping.NoCondition.for_batch_shape(batch_shape),
        )
    model_name = getattr(model, "__name__", type(model).__name__)
    if isinstance(path, seqjtyping.NoCondition) and condition_cls is not seqjtyping.NoCondition:
        raise ValueError(
            f"{model_name} requires a condition path, but an empty condition was supplied."
        )
    
    if not isinstance(path, seqjtyping.NoCondition) and condition_cls is seqjtyping.NoCondition:
        raise ValueError(
            f"{model_name} is unconditional, but a condition path was supplied."
        )
    
    if (
        isinstance(path, seqjtyping.NoCondition)
        and path.batch_shape != batch_shape
    ):
        raise ValueError(
            "NoCondition batch shape must match the expected execution shape: "
            f"got {path.batch_shape}, expected {batch_shape}."
        )
    return path

