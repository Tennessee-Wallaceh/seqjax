from dataclasses import dataclass
import typing

import jax
import jax.numpy as jnp
import jaxtyping

import seqjax.model.typing as seqjtyping
import seqjax.util as util
from seqjax.model.base import BayesianSequentialModel


class InferenceDataset[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
](typing.Protocol):
    """Equal-length batched dataset with leading sequence axis."""

    @property
    def observations(self) -> ObservationT: ...

    @property
    def conditions(self) -> ConditionT: ...

    @property
    def num_sequences(self) -> int: ...

    @property
    def sequence_length(self) -> int: ...

    def sequence(self, idx: int) -> tuple[ObservationT, ConditionT]: ...

@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ObservationDataset[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
]:
    """Concrete equal-length dataset container used by entrypoints."""

    observations: ObservationT
    conditions: ConditionT

    @classmethod
    def from_single_sequence(
        cls,
        observation_path: ObservationT,
        condition_path: ConditionT,
    ) -> typing.Self:
        return cls.from_sequences((observation_path,), (condition_path,))

    @classmethod
    def from_sequences(
        cls,
        observation_paths: tuple[ObservationT, ...],
        condition_paths: tuple[ConditionT, ...],
    ) -> typing.Self:
        if len(observation_paths) != len(condition_paths):
            raise ValueError(
                "Observation and condition collections must have matching lengths. "
                f"Got {len(observation_paths)} observations and "
                f"{len(condition_paths)} conditions."
            )
        if len(observation_paths) == 0:
            raise ValueError("At least one sequence is required for inference.")

        first_length = observation_paths[0].batch_shape[0]
        for idx, (obs, cond) in enumerate(zip(observation_paths, condition_paths)):
            if obs.batch_shape[0] != first_length:
                raise ValueError(
                    "InferenceDataset currently supports only equal-length sequences. "
                    f"Sequence 0 has length {first_length}, sequence {idx} has "
                    f"length {obs.batch_shape[0]}."
                )
            cond_shape = cond.batch_shape
            if len(cond_shape) > 0 and cond_shape[0] != first_length:
                raise ValueError(
                    "Condition sequence length must match observation sequence length. "
                    f"Sequence {idx} has observation length {first_length} "
                    f"and condition length {cond_shape[0]}."
                )

        observations = typing.cast(
            ObservationT,
            jax.tree_util.tree_map(
                lambda *leaves: jnp.stack(leaves, axis=0),
                *observation_paths,
            ),
        )
        conditions = typing.cast(
            ConditionT,
            jax.tree_util.tree_map(
                lambda *leaves: jnp.stack(leaves, axis=0),
                *condition_paths,
            ),
        )
        return cls(observations=observations, conditions=conditions)

    @property
    def num_sequences(self) -> int:
        return int(self.observations.batch_shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.observations.batch_shape[1])

    def sequence(self, idx: int) -> tuple[ObservationT, ConditionT]:
        return (
            typing.cast(ObservationT, util.index_pytree_in_dim(self.observations, idx, 0)),
            typing.cast(ConditionT, util.index_pytree_in_dim(self.conditions, idx, 0)),
        )

    def single_sequence(self) -> tuple[ObservationT, ConditionT]:
        if self.num_sequences != 1:
            raise ValueError(
                "This inference backend currently expects exactly one sequence. "
                f"Received {self.num_sequences} sequences."
            )
        return self.sequence(0)


class InferenceMethod[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](typing.Protocol):
    def __call__(
        self,
        target_posterior: BayesianSequentialModel[
            LatentT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
        ],
        hyperparameters: HyperParametersT,
        key: jaxtyping.PRNGKeyArray,
        dataset: InferenceDataset[ObservationT, ConditionT],
        test_samples: int,
        config: typing.Any,
        tracker: typing.Any = None,
    ) -> tuple[InferenceParametersT, typing.Any]: ...


def adapt_single_sequence_inference[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    f: typing.Callable[..., tuple[InferenceParametersT, typing.Any]],
) -> InferenceMethod[
    LatentT,
    ObservationT,
    ConditionT,
    ParametersT,
    InferenceParametersT,
    HyperParametersT,
]:
    def wrapped(
        target_posterior: BayesianSequentialModel[
            LatentT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
        ],
        hyperparameters: HyperParametersT,
        key: jaxtyping.PRNGKeyArray,
        dataset: InferenceDataset[ObservationT, ConditionT],
        test_samples: int,
        config: typing.Any,
        tracker: typing.Any = None,
    ) -> tuple[InferenceParametersT, typing.Any]:
        if dataset.num_sequences != 1:
            raise NotImplementedError(
                "Sequence-axis-aware inference internals are not yet migrated. "
                f"Received {dataset.num_sequences} sequences."
            )
        observation_path, condition_path = dataset.sequence(0)
        return f(
            target_posterior,
            hyperparameters,
            key,
            observation_path,
            condition_path,
            test_samples,
            config,
            tracker,
        )

    return wrapped


def inference_method(f: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    return f
