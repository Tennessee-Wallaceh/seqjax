"""Protocol typing for sequential models.

``Condition`` and ``Parameter`` are separated. Parameters remain static over time while
conditions vary.
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
from functools import partial

import jax
from jaxtyping import PRNGKeyArray, Scalar

import seqjax.model.typing as seqjtyping

@jax.tree_util.register_dataclass
@dataclass
class FixedLengthHistoryContext[ItemT]:
    """Concrete lag-only history context with bounded length.
    This is basically a wrapper around a tuple that gives some typing utility.
"""

    values: tuple[ItemT, ...]
    length: int = field(metadata=dict(static=True))

    @classmethod
    def from_values(cls, *values: ItemT, length: int) -> typing.Self:
        return cls(values=tuple(values), length=length)
    
    def __getitem__(self, lag_index: int) -> ItemT:
        if not isinstance(lag_index, int):
            raise TypeError("History indices must be integers")
        if lag_index > 0:
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

@jax.tree_util.register_dataclass
@dataclass
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

    prior_order: int
    transition_order: int
    emission_order: int
    observation_dependency: int

    latent_context: typing.Callable[..., LatentContext[LatentT]]
    observation_context: typing.Callable[..., ObservationContext[ObservationT]]
    condition_context: typing.Callable[..., ConditionContext[ConditionT]]

    def prior_sample(
        self,
        key: PRNGKeyArray,
        conditions: ConditionContext[ConditionT],
        parameters: ParametersT,
    ) -> LatentContext[LatentT]: ...

    def prior_log_prob(
        self,
        latent: LatentContext[LatentT],
        conditions: ConditionContext[ConditionT],
        parameters: ParametersT,
    ) -> Scalar: ...

    def transition_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> LatentT: ...

    def transition_log_prob(
        self,
        latent_history: LatentContext[LatentT],
        latent: LatentT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...

    def emission_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT],
        observation_history: ObservationContext[ObservationT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ObservationT: ...

    def emission_log_prob(
        self,
        latent_history: LatentContext[LatentT],
        observation: ObservationT,
        observation_history: ObservationContext[ObservationT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...

def validate_sequential_model[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    model: SequentialModelProtocol[LatentT, ObservationT, ConditionT, ParametersT],
) -> SequentialModelProtocol[LatentT, ObservationT, ConditionT, ParametersT]:
    if model.prior_order < max(model.transition_order, model.emission_order):
        raise ValueError(
            "prior_order must be >= max(transition_order, emission_order), got "
            f"prior_order={model.prior_order} transition_order={model.transition_order} "
            f"emission_order={model.emission_order}"
        )

    return model    

"""
A bayesian model is defined in the following way:
SequentialModel
+ Parameterization(InferenceParameters -> ModelParameters)
+ Prior(InferenceParameters)

For now I package the prior with the Parameterization, since they are quite tightly 
coupled.
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
    
