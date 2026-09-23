"""Protocol typing for sequential models.

``Condition`` and ``Parameter`` are separated. 
Parameters remain static over time while conditions vary.
The assumption is that ``Condition`` will be supplied for every point in the overall sequence.

The primary purpose of model components (e.g Prior, Transition, Emission) is to
pair pure functions for sampling and evaluating log-probabilities.
We can also use typing to expose order information about the dependency structure.

``SequentialModel`` is then used to group model components that operate on the same
``Latent`` and ``Emission`` types.


Specific dependency structures can be expressed by defining custom Prior, Transition
and Emission protocols. The default ones assume first order Markovian structure
(i.e. only depending on the previous latent state) and emissions depending only
on the current latent state.
This requires a fair amount of boilerplate, but allows for nice typing without resorting to
metaclasses.

Alternatives:
- Fully explicit history for generic Transitions etc, model def becomes even more verbose
- Abstract base classes with metaclass magic to enforce structure, but less static typing support
"""

import typing
from dataclasses import dataclass, field

import jax
from jaxtyping import PRNGKeyArray, Scalar

import seqjax.model.typing as seqjtyping

# The context objects slice histories 
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FixedLengthHistoryContext[ItemT]:
    """Lag-only history context containing exactly `length` values."""

    values: tuple[ItemT, ...]
    length: int = field(metadata={"static": True})

    @classmethod
    def from_values(cls, *values: ItemT, length: int) -> typing.Self:
        return cls(values=tuple(values), length=length)

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError("History length must be non-negative")

        if len(self.values) != self.length:
            raise ValueError(
                f"Expected {self.length} history values, "
                f"received {len(self.values)}"
            )

    def __getitem__(self, lag_index: int) -> ItemT:
        if not isinstance(lag_index, int):
            raise TypeError("History indices must be integers")

        if lag_index >= 0:
            raise IndexError("History access is lag-only; use negative indices")

        if -lag_index > len(self.values):
            raise IndexError(
                f"Invalid lag {-lag_index} for history length "
                f"{len(self.values)}"
            )

        return self.values[lag_index]

    def __len__(self) -> int:
        return len(self.values)

    def to_tuple(self) -> tuple[ItemT, ...]:
        return self.values

    def append(self, new_value: ItemT) -> typing.Self:
        """Return a context with `new_value` appended and the oldest removed."""

        new_history = (*self.values, new_value)
        retained_values = new_history[
            len(new_history) - self.length :
        ]

        return type(self).from_values(
            *retained_values,
            length=self.length,
        )

    def __repr__(self) -> str:
        return "<" + "|".join(map(str, self.values)) + ">"
    
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LatentContext[LatentT: seqjtyping.Latent](
    FixedLengthHistoryContext[LatentT],
):
    """Latent history context."""


class ObservedItem[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
](typing.NamedTuple):
    observation: ObservationT
    condition: ConditionT


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservedHistoryContext[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
](
    FixedLengthHistoryContext[
        ObservedItem[ObservationT, ConditionT]
    ],
):
    """Aligned history of past observation-condition pairs."""

    def append_observation(
        self,
        observation: ObservationT,
        condition: ConditionT,
    ) -> typing.Self:
        return self.append(
            ObservedItem(
                observation=observation,
                condition=condition,
            )
        )


# These define the distribution operations
class PriorSampleFn[
    LatentT: seqjtyping.Latent,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        parameters: ParametersT,
    ) -> LatentContext[LatentT]: ...


class PriorLogProbFn[
    LatentT: seqjtyping.Latent,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        latent: LatentContext[LatentT],
        parameters: ParametersT,
    ) -> Scalar: ...


class TransitionSampleFn[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT],
        observation_history: ObservedHistoryContext[ObservationT, ConditionT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> LatentT: ...


class TransitionLogProbFn[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        latent_history: LatentContext[LatentT],
        observation_history: ObservedHistoryContext[ObservationT, ConditionT],
        latent: LatentT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...


class EmissionSampleFn[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT],
        observation_history: ObservedHistoryContext[ObservationT, ConditionT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ObservationT: ...


class EmissionLogProbFn[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        latent_history: LatentContext[LatentT],
        observation: ObservationT,
        observation_history: ObservedHistoryContext[ObservationT, ConditionT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...

class SequentialModelProtocol[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    latent_cls: type[LatentT]
    observation_cls: type[ObservationT]
    parameter_cls: type[ParametersT]
    condition_cls: type[ConditionT]

    transition_latent_order: int
    transition_observation_order: int
    emission_latent_order: int
    emission_observation_order: int

    @property
    def latent_context_length(self) -> int: ...

    @property
    def observation_context_length(self) -> int: ...

    def latent_context(
        self,
        *values: LatentT,
    ) -> LatentContext[LatentT]: ...

    def observed_history_context(
        self,
        *values: ObservedItem[ObservationT, ConditionT],
    ) -> ObservedHistoryContext[ObservationT, ConditionT]: ...

    prior_sample: PriorSampleFn[LatentT, ParametersT]
    prior_log_prob: PriorLogProbFn[LatentT, ParametersT]                        

    transition_sample: TransitionSampleFn[LatentT, ObservationT, ConditionT, ParametersT]
    transition_log_prob: TransitionLogProbFn[LatentT, ObservationT, ConditionT, ParametersT]

    emission_sample: EmissionSampleFn[LatentT, ObservationT, ConditionT, ParametersT]
    emission_log_prob: EmissionLogProbFn[LatentT, ObservationT, ConditionT, ParametersT]                                        


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class SequentialModel[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
):
    latent_cls: type[LatentT] = field(metadata={"static": True})
    observation_cls: type[ObservationT] = field(metadata={"static": True})
    parameter_cls: type[ParametersT] = field(metadata={"static": True})
    condition_cls: type[ConditionT] = field(metadata={"static": True})

    transition_latent_order: int = field(
        default=1,
        metadata={"static": True},
    )
    transition_observation_order: int = field(
        default=0,
        metadata={"static": True},
    )
    emission_latent_order: int = field(
        default=1,
        metadata={"static": True},
    )
    emission_observation_order: int = field(
        default=0,
        metadata={"static": True},
    )

    prior_sample: PriorSampleFn[
        LatentT,
        ConditionT,
        ParametersT,
    ] = field(metadata={"static": True})

    prior_log_prob: PriorLogProbFn[
        LatentT,
        ConditionT,
        ParametersT,
    ] = field(metadata={"static": True})

    transition_sample: TransitionSampleFn[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ] = field(metadata={"static": True})

    transition_log_prob: TransitionLogProbFn[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ] = field(metadata={"static": True})

    emission_sample: EmissionSampleFn[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ] = field(metadata={"static": True})

    emission_log_prob: EmissionLogProbFn[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ] = field(metadata={"static": True})

    def __post_init__(self) -> None:
        order_names = (
            "transition_latent_order",
            "transition_observation_order",
            "emission_latent_order",
            "emission_observation_order",
        )

        for name in order_names:
            value = getattr(self, name)

            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )

            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative, got {value}"
                )

    @property
    def latent_context_length(self) -> int:
        return max(
            self.transition_latent_order,
            self.emission_latent_order,
        )

    @property
    def observation_context_length(self) -> int:
        return max(
            self.transition_observation_order,
            self.emission_observation_order,
        )

    def latent_context(
        self,
        *values: LatentT,
    ) -> LatentContext[LatentT]:
        return LatentContext.from_values(
            *values,
            length=self.latent_context_length,
        )

    def observed_history_context(
        self,
        *values: ObservedItem[ObservationT, ConditionT],
    ) -> ObservedHistoryContext[ObservationT, ConditionT]:
        return ObservedHistoryContext.from_values(
            *values,
            length=self.observation_context_length,
        )


"""
A Bayesian model is defined in the following way:
SequentialModel(ModelParameters)
+ Parameterization(
    InferenceParameters -> ModelParameters,
    HyperParameters
)

The Parameterization also expresses the prior density.
InferenceParameters are unconstrained, 
to enable automatic inference.
"""
class ParameterizationProtocol[
    ParameterT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT,
](typing.Protocol):
    inference_parameter_cls: type[InferenceParametersT]
    hyperparameters: HyperParametersT
    
    def to_model_parameters(
        self,
        inference_parameters: InferenceParametersT,
    ) -> ParameterT:
        ...

    def from_model_parameters(
        self,
        model_parameters: ParameterT,
    ) -> InferenceParametersT:
        ...

    def log_prob(self, inference_parameters: InferenceParametersT) -> Scalar:
        ...

    def sample(self, key: PRNGKeyArray) -> InferenceParametersT:
        ...


class BayesianSequentialModelProtocol[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT,
](typing.Protocol):
    target: SequentialModelProtocol[LatentT, ObservationT, ConditionT, ParameterT]
    parameterization: ParameterizationProtocol[ParameterT, InferenceParametersT, HyperParametersT]


def validate_bayesian_model[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT,
](
    model: BayesianSequentialModelProtocol[
        LatentT, ObservationT, ConditionT, ParametersT, InferenceParametersT, HyperParametersT
    ],
) -> BayesianSequentialModelProtocol[
    LatentT, ObservationT, ConditionT, ParametersT, InferenceParametersT, HyperParametersT
]:
    return model
    
