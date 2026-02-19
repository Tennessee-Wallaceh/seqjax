from abc import abstractmethod
import typing
from dataclasses import dataclass

import jax
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
    ) -> tuple[
        TargetStructT,
        jaxtyping.Scalar,
    ]: ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LatentContext[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    HiddenT: jaxtyping.Array,
]:
    """
    The latent context offers global (y, c, theta) context related to the 
    sample draw. We also support an embedded context, 

    It also supports information being passed for specific points in the sequence, 
    via the sequence_... options. 

    Various approximations will make use of different elements of this context.
    The intention is that unused computations will be compiled away. 
    """
    observation_context: ObservationT
    condition_context: ConditionT
    parameter_context: ParameterT
    embedded_context: jaxtyping.Array

    sequence_embedded_context: HiddenT

    @classmethod
    def build_from_sequence_context(
        cls, 
        sequence_embedded_context: HiddenT,
        observations: ObservationT,
        conditions: ConditionT,
        parameters: ParameterT,
    ):
        return cls(
            observation_context=observations,
            condition_context=conditions,
            parameter_context=parameters,
            embedded_context=observations.ravel(),
            sequence_embedded_context=sequence_embedded_context,
        )

class Embedder[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
](eqx.Module):
    """
    Maps observation sequence to a context vector for
    each point in the batch
    """
    context_dimension: int

    @abstractmethod
    def embed(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        parameters: ParameterT,
    ) -> LatentContext[
        ObservationT,
        ConditionT,
        ParameterT,
        jaxtyping.Array,
    ]: ...


class AmortizedVariationalApproximation[
    TargetStructT: seqjtyping.Packable,
](VariationalApproximation[
    TargetStructT,
    LatentContext,
]):
    batch_length: int
    buffer_length: int

class UnconditionalVariationalApproximation[
    TargetStructT: seqjtyping.Packable,
](VariationalApproximation[TargetStructT, None]):
    @abstractmethod
    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: None = None,
    ) -> tuple[TargetStructT, jaxtyping.Scalar]: ...


class VariationalApproximationFactory[
    TargetStructT: seqjtyping.Packable,
    ConditionT,
](typing.Protocol):
    def __call__(
        self, target_struct_cls: type[TargetStructT]
    ) -> VariationalApproximation[TargetStructT, ConditionT]: ...