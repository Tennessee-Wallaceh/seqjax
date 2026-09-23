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
@dataclass
class FixedLengthHistoryContext[ItemT]:
    """Lag-only history context with a fixed target length.

    The context may contain fewer values during initialization, but never more
    than `length`. New values are appended and the oldest values discarded once
    the target length is reached.
    """

    values: tuple[ItemT, ...]
    length: int = field(metadata=dict(static=True))

    @classmethod
    def from_values(cls, *values: ItemT, length: int) -> typing.Self:
        return cls(values=tuple(values), length=length)
    
    def __getitem__(self, lag_index: int) -> ItemT:
        if not isinstance(lag_index, int):
            raise TypeError("History indices must be integers")
        if lag_index >= 0:
            raise IndexError("History access is lag-only; use negative indices")
        if -lag_index > len(self.values):
            raise IndexError(
                f"Invalid lag {-lag_index} for history length {len(self.values)}"
            )
        return self.values[lag_index]

    def to_tuple(self) -> tuple[ItemT, ...]:
        return self.values
    
    def __repr__(self) -> str:
        return "<" + "|".join([str(val) for val in self.values]) + ">"

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError("History length must be non-negative")
        if len(self.values) != self.length:
            raise ValueError(
                f"Expected {self.length} history values, "
                f"received {len(self.values)}"
            )
        
    def append(self, new_value: ItemT) -> typing.Self:
        """Return a new history with the oldest value replaced."""
        new_history = (*self.values, new_value)
        new_context = new_history[
            len(new_history) - self.length:
        ]

        return type(self).from_values(
            *new_context,
            length=self.length,
        )
    
@jax.tree_util.register_dataclass
class LatentContext[LatentT: seqjtyping.Latent](
    FixedLengthHistoryContext[LatentT],
):
    """Concrete latent history context."""


@jax.tree_util.register_dataclass
class ObservationContext[ObservationT: seqjtyping.Observation](
    FixedLengthHistoryContext[ObservationT],
):
    """Concrete observation history context."""


@jax.tree_util.register_dataclass
class ConditionContext[ConditionT: seqjtyping.Condition](
    FixedLengthHistoryContext[ConditionT],
):
    """Concrete condition history context."""

# These define the distribution oper
class PriorSampleFn[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        conditions: ConditionContext[ConditionT],
        parameters: ParametersT,
    ) -> LatentContext[LatentT]: ...


class PriorLogProbFn[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        latent: LatentContext[LatentT],
        conditions: ConditionContext[ConditionT],
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
        observation_history: ObservationContext[ObservationT],
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
        observation_history: ObservationContext[ObservationT],
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
        observation_history: ObservationContext[ObservationT],
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
        observation_history: ObservationContext[ObservationT],
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
    def prior_latent_order(self) -> int: ...

    @property
    def observation_context_length(self) -> int: ...

    def latent_context(self, *values: LatentT) -> LatentContext[LatentT]: ...

    def observation_context(self, *values: ObservationT) -> ObservationContext[ObservationT]: ...

    def condition_context(self, *values: ConditionT) -> ConditionContext[ConditionT]: ...

    prior_sample: PriorSampleFn[LatentT, ConditionT, ParametersT]
    prior_log_prob: PriorLogProbFn[LatentT, ConditionT, ParametersT]                        

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
    ] = field(
        metadata={"static": True},
        repr=False,
    )

    prior_log_prob: PriorLogProbFn[
        LatentT,
        ConditionT,
        ParametersT,
    ] = field(
        metadata={"static": True},
        repr=False,
    )

    transition_sample: TransitionSampleFn[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ] = field(
        metadata={"static": True},
        repr=False,
    )

    transition_log_prob: TransitionLogProbFn[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ] = field(
        metadata={"static": True},
        repr=False,
    )

    emission_sample: EmissionSampleFn[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ] = field(
        metadata={"static": True},
        repr=False,
    )

    emission_log_prob: EmissionLogProbFn[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ] = field(
        metadata={"static": True},
        repr=False,
    )

    @property
    def prior_latent_order(self) -> int:
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
            length=self.prior_order,
        )

    def observation_context(
        self,
        *values: ObservationT,
    ) -> ObservationContext[ObservationT]:
        return ObservationContext.from_values(
            *values,
            length=self.observation_context_length,
        )

    def condition_context(
        self,
        *values: ConditionT,
    ) -> ConditionContext[ConditionT]:
        return ConditionContext.from_values(
            *values,
            length=0,
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
    
