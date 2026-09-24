
import typing

import seqjax.model.typing as seqjtyping
from seqjax.model import (
    interface as model_interface,
)


def normalize_observation_history[
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
    observation_history: (
        model_interface.ObservedHistoryContext[
            ObservationT,
            ConditionT,
        ]
        | None
    ),
) -> model_interface.ObservedHistoryContext[
    ObservationT,
    ConditionT,
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

