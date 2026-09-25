"""Protocols and dataclasses for sequential models.

A sequential model pairs sampling and log-probability functions for its prior,
transition, and emission distributions. Model parameters are shared across
time; a condition is supplied for each transition and its associated
observation. The prior is sampled before those steps.

Dependency orders count past values needed by transition and emission
functions. A current latent is passed separately to emission functions, so
an emission depending only on that latent has latent order zero. Observation
histories contain observation-condition pairs. The model computes each
context length as the maximum order required by its functions, and context
construction checks the length at runtime.

Context lengths are also type parameters, allowing protocols to require
matching histories across model operations. Python's type system cannot
derive those parameters from the maximum of the order fields, so that
relationship remains a runtime responsibility.

A Bayesian sequential model pairs a sequential model with a
parameterization. The parameterization maps unconstrained inference
parameters to model parameters and defines their prior distribution.
"""

import typing
from dataclasses import dataclass, field

import jax
from jaxtyping import PRNGKeyArray, Scalar

import seqjax.model.typing as seqjtyping

# The context objects slice histories 
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FixedLengthHistoryContext[ItemT, LengthT: int]:
    """Lag-only history context containing exactly `length` values."""

    values: tuple[ItemT, ...]
    length: LengthT = field(metadata={"static": True})

    @classmethod
    def from_values(cls, *values: ItemT, length: LengthT) -> typing.Self:
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
class LatentContext[LatentT: seqjtyping.Latent, LengthT: int](
    FixedLengthHistoryContext[LatentT, LengthT],
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
    LengthT: int,
](
    FixedLengthHistoryContext[
        ObservedItem[ObservationT, ConditionT],
        LengthT
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
    LatentContextLength: int
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        parameters: ParametersT,
    ) -> LatentContext[LatentT, LatentContextLength]: ...


class PriorLogProbFn[
    LatentT: seqjtyping.Latent,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int
](typing.Protocol):
    def __call__(
        self,
        latent: LatentContext[LatentT, LatentContextLength],
        parameters: ParametersT,
    ) -> Scalar: ...


class TransitionSampleFn[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT, LatentContextLength],
        parameters: ParametersT,
        condition: ConditionT,
        observation_history: ObservedHistoryContext[
            ObservationT, 
            ConditionT, 
            ObservationContextLength
        ],
    ) -> LatentT: ...


class TransitionLogProbFn[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
](typing.Protocol):
    def __call__(
        self,
        latent: LatentT,
        latent_history: LatentContext[LatentT, LatentContextLength],
        parameters: ParametersT,
        condition: ConditionT,
        observation_history: ObservedHistoryContext[
            ObservationT, 
            ConditionT,
            ObservationContextLength,
        ],
    ) -> Scalar: ...


class EmissionSampleFn[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        latent: LatentT,
        parameters: ParametersT,
        condition: ConditionT,
        latent_history: LatentContext[LatentT, LatentContextLength],
        observation_history: ObservedHistoryContext[
            ObservationT, 
            ConditionT,
            ObservationContextLength
        ],
    ) -> ObservationT: ...


class EmissionLogProbFn[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
](typing.Protocol):
    def __call__(
        self,
        observation: ObservationT,
        latent: LatentT,
        parameters: ParametersT,
        condition: ConditionT,
        latent_history: LatentContext[LatentT, LatentContextLength],
        observation_history: ObservedHistoryContext[
            ObservationT, 
            ConditionT,
            ObservationContextLength
        ],
    ) -> Scalar: ...


class SequentialModelProtocol[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
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
    ) -> LatentContext[LatentT, LatentContextLength]: ...

    def observed_history_context(
        self,
        *values: ObservedItem[ObservationT, ConditionT],
    ) -> ObservedHistoryContext[
        ObservationT, 
        ConditionT, 
        ObservationContextLength
    ]: ...

    prior_sample: PriorSampleFn[
        LatentT, 
        ParametersT, 
        LatentContextLength,
    ]
    prior_log_prob: PriorLogProbFn[
        LatentT, 
        ParametersT, 
        LatentContextLength,
    ]                        

    transition_sample: TransitionSampleFn[
        LatentT, 
        ObservationT, 
        ConditionT, 
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ]
    transition_log_prob: TransitionLogProbFn[
        LatentT, 
        ObservationT, 
        ConditionT, 
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ]

    emission_sample: EmissionSampleFn[
        LatentT, 
        ObservationT, 
        ConditionT,
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ]
    emission_log_prob: EmissionLogProbFn[
        LatentT, 
        ObservationT, 
        ConditionT, 
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ]                                        


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class SequentialModel[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
](
    SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
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
        default=0,
        metadata={"static": True},
    )
    emission_observation_order: int = field(
        default=0,
        metadata={"static": True},
    )

    prior_sample: PriorSampleFn[
        LatentT, 
        ParametersT, 
        LatentContextLength,
    ] = field(metadata={"static": True})

    prior_log_prob: PriorLogProbFn[
        LatentT, 
        ParametersT, 
        LatentContextLength,
    ] = field(metadata={"static": True})

    transition_sample: TransitionSampleFn[
        LatentT, 
        ObservationT, 
        ConditionT, 
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ] = field(metadata={"static": True})

    transition_log_prob: TransitionLogProbFn[
        LatentT, 
        ObservationT, 
        ConditionT, 
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ] = field(metadata={"static": True})

    emission_sample: EmissionSampleFn[
        LatentT, 
        ObservationT, 
        ConditionT,
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
    ] = field(metadata={"static": True})

    emission_log_prob: EmissionLogProbFn[
        LatentT, 
        ObservationT, 
        ConditionT, 
        ParametersT,
        LatentContextLength,
        ObservationContextLength,
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
    def latent_context_length(self) -> LatentContextLength:
        return typing.cast(
            LatentContextLength,
            max(
                self.transition_latent_order,
                self.emission_latent_order,
            )
        )

    @property
    def observation_context_length(self) -> ObservationContextLength:
        return typing.cast(
            ObservationContextLength,
            max(
                self.transition_observation_order,
                self.emission_observation_order,
            )
        )

    def latent_context(
        self,
        *values: LatentT,
    ) -> LatentContext[LatentT, LatentContextLength]:
        return LatentContext.from_values(
            *values,
            length=self.latent_context_length,
        )

    def observed_history_context(
        self,
        *values: ObservedItem[ObservationT, ConditionT],
    ) -> ObservedHistoryContext[ObservationT, ConditionT, ObservationContextLength]:
        return ObservedHistoryContext.from_values(
            *values,
            length=self.observation_context_length,
        )


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
    LatentContextLength: int,
    ObservationContextLength: int,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT,
](typing.Protocol):
    target: SequentialModelProtocol[
        LatentT, 
        ObservationT, 
        ConditionT, 
        ParameterT,
        LatentContextLength,
        ObservationContextLength,
    ]
    parameterization: ParameterizationProtocol[ParameterT, InferenceParametersT, HyperParametersT]
    

@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class BayesianSequentialModel[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    LatentContextLength: int,
    ObservationContextLength: int,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT,
](
    BayesianSequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParameterT,
        LatentContextLength,
        ObservationContextLength,
        InferenceParametersT,
        HyperParametersT,
    ],
):
    target: SequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParameterT,
        LatentContextLength,
        ObservationContextLength,
    ]
    parameterization: ParameterizationProtocol[
        ParameterT,
        InferenceParametersT,
        HyperParametersT,
    ]