from abc import abstractmethod
import typing
from dataclasses import dataclass, field
from seqjax.model.interface import BayesianSequentialModelProtocol
import jax
import jax.numpy as jnp
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
    def from_sequence_context_dims(
        cls, 
        target_posterior: BayesianSequentialModelProtocol[
            typing.Any,
            ObservationT,
            ConditionT,
            typing.Any,
            InferenceParameterT,
            typing.Any,
        ], 
        sample_length: int
    ) -> tuple[int, int, int, int]:
        return (
            target_posterior.target.observation_cls.flat_dim * sample_length,
            target_posterior.target.condition_cls.flat_dim * sample_length,
            target_posterior.parameterization.inference_parameter_cls.flat_dim,
            target_posterior.target.observation_cls.flat_dim * sample_length
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
            embedded_context=observations.ravel(),
            sequence_embedded_context=sequence_embedded_context,
        )

    @classmethod
    def from_sequence_and_embedded_dims(
        cls,
        target_posterior: BayesianSequentialModelProtocol[
            typing.Any,
            ObservationT,
            ConditionT,
            typing.Any,
            InferenceParameterT,
            typing.Any,
        ], 
        sample_length: int
    ) -> tuple[int, int, int]:
        return (
            target_posterior.target.observation_cls.flat_dim * sample_length,
            target_posterior.target.condition_cls.flat_dim * sample_length,
            target_posterior.parameterization.inference_parameter_cls.flat_dim,
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
            embedded_context=embedded_context,
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
    target_posterior: BayesianSequentialModelProtocol
    sample_length: int
    sequence_length: int
    observation_context_dim: int = field(init=False)
    condition_context_dim: int = field(init=False)
    parameter_context_dim: int = field(init=False)
    embedded_context_dim: int = field(init=False)
    sequence_embedded_context_dim: int = field(init=False) # per step

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
