from abc import abstractmethod
import typing
from dataclasses import dataclass

from seqjax.model.interface import SequentialModelProtocol
from .norm import InputNormalization

import jax
import equinox as eqx
import jaxtyping
import seqjax.model.typing as seqjtyping


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class LatentContextDims:
    observation_context_dim: int
    condition_context_dim: int
    parameter_context_dim: int
    flat_features_dim: int
    sequence_features_dim: int

    @classmethod
    def from_sequence_features_dim(
        cls,
        target: SequentialModelProtocol,
        parameter_cls: type[seqjtyping.Parameters],
        sample_length: int,
        sequence_features_dim: int,
    ) -> typing.Self:
        return cls(
            observation_context_dim=target.observation_cls.flat_dim * sample_length,
            condition_context_dim=target.condition_cls.flat_dim,
            parameter_context_dim=parameter_cls.flat_dim,
            flat_features_dim=target.observation_cls.flat_dim * sample_length,
            sequence_features_dim=sequence_features_dim,
        )

    @classmethod
    def from_flat_and_sequence_feature_dims(
        cls,
        target: SequentialModelProtocol,
        parameter_cls: type[seqjtyping.Parameters],
        sample_length: int,
        flat_features_dim: int,
        sequence_features_dim: int,
    ) -> typing.Self:
        return cls(
            observation_context_dim=target.observation_cls.flat_dim * sample_length,
            condition_context_dim=target.condition_cls.flat_dim,
            parameter_context_dim=parameter_cls.flat_dim,
            flat_features_dim=flat_features_dim,
            sequence_features_dim=sequence_features_dim,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LatentContext[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    InferenceParameterT: seqjtyping.Parameters,
    HiddenT: jaxtyping.Array,
]:
    """Raw model context and inference features for a latent-path draw.

    The observation, condition, and parameter contexts retain their original
    structured values for model-based operations inside an approximation.
    ``flat_features`` provides a single global feature vector, while
    ``sequence_features`` provides one feature vector per sequence position.
    Each embedder decides how those two feature views are constructed.
    """

    observation_context: ObservationT
    condition_context: ConditionT
    parameter_context: InferenceParameterT
    flat_features: jaxtyping.Array
    sequence_features: jaxtyping.Array

    @classmethod
    def spec(
        cls,
        *,
        observation_context: int | None,
        condition_context: int | None,
        parameter_context: int | None,
        flat_features: int | None,
        sequence_features: int | None,
    ) -> typing.Self:
        return cls(
            observation_context=observation_context,
            condition_context=condition_context,
            parameter_context=parameter_context,
            flat_features=flat_features,
            sequence_features=sequence_features,
        )

    @classmethod
    def build_from_sequence_features(
        cls,
        sequence_features: HiddenT,
        observations: ObservationT,
        conditions: ConditionT,
        parameters: InferenceParameterT,
    ) -> typing.Self:
        return cls(
            observation_context=observations,
            condition_context=conditions,
            parameter_context=parameters,
            flat_features=observations.ravel().flatten(),
            sequence_features=sequence_features,
        )

    @classmethod
    def build_from_flat_and_sequence_features(
        cls,
        sequence_features: HiddenT,
        flat_features: jaxtyping.Array,
        observations: ObservationT,
        conditions: ConditionT,
        parameters: InferenceParameterT,
    ) -> typing.Self:
        return cls(
            observation_context=observations,
            condition_context=conditions,
            parameter_context=parameters,
            flat_features=flat_features.ravel().flatten(),
            sequence_features=sequence_features,
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
    """Builds flat and per-position inference features for a sequence."""

    target: SequentialModelProtocol
    parameter_cls: type[InferenceParameterT]
    sample_length: int
    sequence_length: int
    latent_context_dims: LatentContextDims
    normalization: InputNormalization

    def _normalize_inputs(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        parameters: InferenceParameterT,
        state: typing.Any,
        *,
        reduce_axes: tuple[str, ...],
        training: bool,
    ) -> tuple[ObservationT, jaxtyping.Array, jaxtyping.Array, typing.Any]:
        observation_values = observations.ravel()
        condition_values = conditions.ravel()
        parameter_values = parameters.ravel()
        if observation_values.shape != (
            self.sample_length,
            self.target.observation_cls.flat_dim,
        ):
            raise ValueError(
                "observations must have shape "
                f"({self.sample_length}, {self.target.observation_cls.flat_dim}), "
                f"got {observation_values.shape}"
            )
        if condition_values.shape != (
            self.sample_length,
            self.target.condition_cls.flat_dim,
        ):
            raise ValueError(
                "conditions must have shape "
                f"({self.sample_length}, {self.target.condition_cls.flat_dim}), "
                f"got {condition_values.shape}"
            )
        if parameter_values.shape != (self.parameter_cls.flat_dim,):
            raise ValueError(
                "parameters must have shape "
                f"({self.parameter_cls.flat_dim},), got {parameter_values.shape}"
            )
        observation_values, condition_values, parameter_values, state = (
            self.normalization.normalize(
                observation_values,
                condition_values,
                parameter_values,
                state,
                reduce_axes=reduce_axes,
                training=training,
            )
        )
        return (
            observations.unravel(observation_values),
            condition_values,
            parameter_values,
            state,
        )

    def _augment_sequence_features(
        self,
        sequence_features: jaxtyping.Array,
        normalized_conditions: jaxtyping.Array,
    ) -> jaxtyping.Array:
        if self.normalization.include_condition:
            return jax.numpy.concatenate(
                [sequence_features, normalized_conditions], axis=-1
            )
        return sequence_features

    def _augment_flat_features(
        self,
        flat_features: jaxtyping.Array,
        normalized_parameters: jaxtyping.Array,
    ) -> jaxtyping.Array:
        if self.normalization.include_parameter:
            return jax.numpy.concatenate(
                [flat_features.ravel(), normalized_parameters.ravel()]
            )
        return flat_features

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
