"""Declarative condition alignment for sequential model execution."""

from dataclasses import dataclass
import typing

import seqjax.model.typing as seqjtyping
from seqjax import util
from seqjax.model.interface import ConditionContext


class ConditionLayoutProtocol[ConditionT: seqjtyping.Condition](typing.Protocol):
    """Map a packed condition path onto prior, transition, and emission calls."""

    def prepare(
        self, model: typing.Any, path: ConditionT, observation_count: int
    ) -> "PreparedConditions[ConditionT]": ...
    def slice_window(self, path: ConditionT, start: int, length: int) -> ConditionT: ...


class SupportsConditionLayout(typing.Protocol):
    """Optional model capability for selecting a non-default layout."""

    condition_layout: ConditionLayoutProtocol


@dataclass(frozen=True)
class PreparedConditions[ConditionT: seqjtyping.Condition]:
    """Conditions prepared for one complete sequential-model execution."""

    prior: ConditionContext[ConditionT]
    initial_emission: ConditionT
    transitions: ConditionT
    recurrent_emissions: ConditionT
    emissions: ConditionT


@dataclass(frozen=True)
class StepAlignedConditions:
    """Conditions indexed by emission step and transition destination.

    ``condition[t]`` is used by emission ``t`` and, for ``t > 0``, by the
    transition into the latent state at step ``t``.  Prior conditioning is an
    explicit prefix length and is independent of the number of initial latent
    states.
    """

    prior_condition_count: int | None = None

    def _require_length(self, path: seqjtyping.Condition, required: int) -> None:
        if isinstance(path, seqjtyping.NoCondition):
            return
        actual = path.batch_shape[0]
        if actual < required:
            raise ValueError(
                "Condition path is too short for the model layout: "
                f"got {actual}, expected at least {required}."
            )

    def prepare(
        self,
        model: typing.Any,
        path: seqjtyping.Condition,
        observation_count: int,
    ) -> PreparedConditions:
        if observation_count < 1:
            raise ValueError(
                "Condition preparation requires at least one observation; "
                f"got observation_count={observation_count}."
            )
        if isinstance(path, seqjtyping.NoCondition):
            return PreparedConditions(
                prior=model.condition_context(()),
                initial_emission=path,
                transitions=path,
                recurrent_emissions=path,
                emissions=path,
            )
        count = (
            model.prior_order
            if self.prior_condition_count is None
            else self.prior_condition_count
        )
        self._require_length(path, max(count, observation_count))
        prior = model.condition_context(
            tuple(util.index_pytree(path, index) for index in range(count))
        )
        emissions = util.slice_pytree(path, 0, observation_count)
        recurrent = util.slice_pytree(path, 1, observation_count)
        return PreparedConditions(
            prior=prior,
            initial_emission=util.index_pytree(emissions, 0),
            transitions=recurrent,
            recurrent_emissions=recurrent,
            emissions=emissions,
        )

    def slice_window(self, path: seqjtyping.Condition, start: int, length: int):
        if isinstance(path, seqjtyping.NoCondition):
            return path
        return util.dynamic_slice_pytree(path, start, length)


DEFAULT_CONDITION_LAYOUT = StepAlignedConditions()


def layout_for(model: typing.Any) -> ConditionLayoutProtocol:
    return typing.cast(
        ConditionLayoutProtocol,
        getattr(model, "condition_layout", DEFAULT_CONDITION_LAYOUT),
    )
