from abc import abstractmethod
import typing
from dataclasses import dataclass

from seqjax.model.interface import SequentialModelProtocol

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
        state: typing.Any = None,
    ) -> tuple[
        TargetStructT,
        jaxtyping.Scalar,
        typing.Any,
    ]: ...

@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class LatentContextDims:
    observation_context_dim: int
    condition_context_dim: int
    parameter_context_dim: int
    embedded_context_dim: int
    sequence_embedded_context_dim: int

    @classmethod
    def from_sequence_context_dims(
        cls, 
        target: SequentialModelProtocol,
        parameter_cls: type[seqjtyping.Parameters],
        sample_length: int,
        per_step_dim: int,
    ) -> tuple[int, int, int, int]:
        return cls(
            observation_context_dim=target.observation_cls.flat_dim * sample_length,
            condition_context_dim=target.condition_cls.flat_dim * sample_length,
            parameter_context_dim=parameter_cls.flat_dim,
            embedded_context_dim=target.observation_cls.flat_dim * sample_length,
            sequence_embedded_context_dim=per_step_dim
        )


    @classmethod
    def from_sequence_and_embedded_dims(
        cls,
        target: SequentialModelProtocol,
        parameter_cls: type[seqjtyping.Parameters],
        sample_length: int,
        embedded_dim: int,
        per_step_dim: int,
    ) -> tuple[int, int, int]:
        return cls(
            observation_context_dim=target.observation_cls.flat_dim * sample_length,
            condition_context_dim=target.condition_cls.flat_dim * sample_length,
            parameter_context_dim=parameter_cls.flat_dim,
            embedded_context_dim=embedded_dim,
            sequence_embedded_context_dim=per_step_dim
        )
    



@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LatentContext[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    InferenceParameterT: seqjtyping.Parameters,
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
    parameter_context: InferenceParameterT
    embedded_context: jaxtyping.Array
    sequence_embedded_context: jaxtyping.Array

    @classmethod
    def spec(
        cls,
        *,
        observation_context: int | None,
        condition_context: int | None,
        parameter_context: int | None,
        embedded_context: int | None,
        sequence_embedded_context: int | None,
    ):
        return cls(
            observation_context,
            condition_context,
            parameter_context,
            embedded_context,
            sequence_embedded_context,
        )
    
    @classmethod
    def build_from_sequence_context(
        cls, 
        sequence_embedded_context: HiddenT,
        observations: ObservationT,
        conditions: ConditionT,
        parameters: InferenceParameterT,

    ):  
        return cls(
            observation_context=observations,
            condition_context=conditions,
            parameter_context=parameters,
            embedded_context=observations.ravel().flatten(),
            sequence_embedded_context=sequence_embedded_context,
        )

    @classmethod
    def build_from_sequence_and_embedded(
        cls, 
        sequence_embedded_context: HiddenT,
        embedded_context: jaxtyping.Array,
        observations: ObservationT,
        conditions: ConditionT,
        parameters: InferenceParameterT,
    ):  
        return cls(
            observation_context=observations,
            condition_context=conditions,
            parameter_context=parameters,
            embedded_context=embedded_context.ravel().flatten(),
            sequence_embedded_context=sequence_embedded_context,
        )



class SequenceAggregator(typing.Protocol):
    @property
    def output_dim(self) -> int: ...

    def __call__(
        self,
        sequence_features: jaxtyping.Array,
        observations: seqjtyping.Observation,
    ) -> jaxtyping.Array: ...


class Embedder[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    InferenceParameterT: seqjtyping.Parameters,
](eqx.Module):
    """
    Maps observation sequence to various embeddings.
    """
    target: SequentialModelProtocol
    parameter_cls: InferenceParameterT
    sample_length: int
    sequence_length: int
    latent_context_dims: LatentContextDims

    @abstractmethod
    def embed(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        parameters: InferenceParameterT,
        state: typing.Any = None,
        *,
        sequence_start: None | int = None,
        reduce_axes: tuple[str, ...] = (),
        training: bool = False,
    ) -> tuple[
        LatentContext[
            ObservationT,
            ConditionT,
            InferenceParameterT,
            jaxtyping.Array,
        ],
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
