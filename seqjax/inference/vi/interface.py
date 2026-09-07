from abc import abstractmethod
import typing

from seqjax.inference.vi.embedder.interface import LatentContext

import equinox as eqx
import jaxtyping
import seqjax.model.typing as seqjtyping


class VariationalApproximation[
    TargetStructT: seqjtyping.Packable,
    ConditionT,
](eqx.Module):
    target_struct_cls: type[TargetStructT] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)

    @abstractmethod
    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: ConditionT,
        state: typing.Any = None,
    ) -> tuple[
        TargetStructT,
        jaxtyping.Scalar,
        typing.Any,
    ]: ...

class AmortizedVariationalApproximation[
    TargetStructT: seqjtyping.Packable,
](VariationalApproximation[
    TargetStructT,
    LatentContext,
]):
    sample_length: int


class UnconditionalVariationalApproximation[
    TargetStructT: seqjtyping.Packable,
](VariationalApproximation[TargetStructT, None]):
    @abstractmethod
    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: None = None,
        state: typing.Any = None,
    ) -> tuple[TargetStructT, jaxtyping.Scalar, typing.Any]: ...


class VariationalApproximationFactory[
    TargetStructT: seqjtyping.Packable,
    ConditionT,
](typing.Protocol):
    def __call__(
        self, target_struct_cls: type[TargetStructT]
    ) -> VariationalApproximation[TargetStructT, ConditionT]: ...
