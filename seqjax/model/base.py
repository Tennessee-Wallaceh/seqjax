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

from typing import Callable
from functools import partial
from abc import ABC, abstractmethod
import typing
from types import SimpleNamespace
from dataclasses import dataclass, field

import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar
import jax.numpy as jnp

import seqjax.model.typing as seqjtyping


class FixedLengthHistoryContext[ItemT]:
    """Concrete lag-only history context with bounded length.

    ``__getitem__`` accepts only negative integer indices to enforce lag semantics.
    """

    values: tuple[ItemT, ...]
    max_length: int

    def __init__(self, *values: ItemT, max_length: int):
        if len(values) == 1 and isinstance(values[0], tuple):
            values = typing.cast(tuple[ItemT, ...], values[0])
        self.values = tuple(values)
        self.max_length = max_length

        if self.max_length < 0:
            raise ValueError("max_length must be >= 0")
        if len(self.values) > self.max_length:
            raise ValueError(
                "values length cannot exceed max_length: "
                f"len(values)={len(self.values)} max_length={self.max_length}"
            )

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

    @property
    def current_length(self) -> int:
        return len(self.values)

    def append(self, item: ItemT) -> typing.Self:
        if self.max_length == 0:
            return type(self)(max_length=0)
        next_values = (*self.values, item)
        next_values = next_values[-self.max_length :]
        return type(self)(*next_values, max_length=self.max_length)

    def to_tuple(self) -> tuple[ItemT, ...]:
        return self.values


class LatentContext[LatentT: seqjtyping.Latent](
    FixedLengthHistoryContext[LatentT],
):
    """Concrete latent history context."""


class ObservationContext[ObservationT: seqjtyping.Observation](
    FixedLengthHistoryContext[ObservationT],
):
    """Concrete observation history context."""


class ConditionContext[ConditionT: seqjtyping.Condition](
    FixedLengthHistoryContext[ConditionT],
):
    """Concrete condition history context."""



LatentContextType = LatentContext
ObservationContextType = ObservationContext
ConditionContextType = ConditionContext

class ParameterPrior[
    ParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](typing.Protocol):
    """Parameter prior specified as utility for specifying Bayesian models."""

    sample: Callable[
        [
            PRNGKeyArray,
            HyperParametersT,
        ],
        ParametersT,
    ]

    log_prob: Callable[
        [
            ParametersT,
            HyperParametersT,
        ],
        Scalar,
    ]


PriorConditionsT = typing.TypeVar("PriorConditionsT")
ParametersT = typing.TypeVar("ParametersT", bound=seqjtyping.Parameters)


class PriorSample[PriorConditionsT, ParametersT, PriorLatentT](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        conditions: PriorConditionsT,
        parameters: ParametersT,
    ) -> PriorLatentT: ...


class PriorLogProb[PriorConditionsT, ParametersT, PriorLatentT](typing.Protocol):
    def __call__(
        self,
        latent: PriorLatentT,
        conditions: PriorConditionsT,
        parameters: ParametersT,
    ) -> Scalar: ...


@dataclass(frozen=True)
class Prior[
    PriorLatentT,
    PriorConditionsT,
    ParametersT: seqjtyping.Parameters,
]:
    """Prior must define density + sampling up to the start state t=0.
    As such it receives conditions up to t=0, length corresponding to order.

    This could be over a number of latents if the dependency structure of the model requires.

    For example if the transition is (x_t1, x_t2) -> x_t3 then the prior must produce 2 latents (x_t-1, x_t0).
    Even if the transition is (x_t1,) -> x_t2, if emission is (x_t1, x_t2) -> y_t2 then
    the prior must specify p(x_t-1, x_t0), so that p(y_t0 | x_t-1, x_t0) can be evaluated.
    """

    order: int
    sample: PriorSample[PriorConditionsT, ParametersT, PriorLatentT]
    log_prob: PriorLogProb[PriorConditionsT, ParametersT, PriorLatentT]


class TransitionSample[LatentHistoryT, ConditionT, ParametersT, LatentT](
    typing.Protocol
):
    def __call__(
        self,
        key: PRNGKeyArray,
        latent_history: LatentHistoryT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> LatentT: ...


class TransitionLogProb[LatentHistoryT, ConditionT, ParametersT, LatentT](
    typing.Protocol
):
    def __call__(
        self,
        latent_history: LatentHistoryT,
        latent: LatentT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...


@dataclass(frozen=True)
class Transition[
    LatentHistoryT,
    LatentT,
    ConditionT,
    ParametersT: seqjtyping.Parameters,
]:
    order: int
    sample: TransitionSample[LatentHistoryT, ConditionT, ParametersT, LatentT]
    log_prob: TransitionLogProb[LatentHistoryT, ConditionT, ParametersT, LatentT]


class GaussianLocScale1[LatentT, ConditionT, ParametersT](typing.Protocol):
    """Return (loc_x, scale_x) for x-space Gaussian transition."""

    def __call__(
        self,
        latent_history: tuple[LatentT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> tuple[Scalar, Scalar]: ...


@dataclass(frozen=True)
class GaussianLocScaleTransition[
    LatentHistoryT,
    LatentT: seqjtyping.Latent,
    ConditionT,
    ParametersT: seqjtyping.Parameters,
](Transition):
    """
    Defines a Transition via a location scale fcn.
    """

    loc_scale: GaussianLocScale1[LatentT, ConditionT, ParametersT]
    latent_t: type[LatentT]

    order: int = field(init=False)
    sample: TransitionSample[LatentHistoryT, ConditionT, ParametersT, LatentT] = field(
        init=False
    )
    log_prob: TransitionLogProb[LatentHistoryT, ConditionT, ParametersT, LatentT] = (
        field(init=False)
    )

    def __post_init__(self):
        def sample(
            key: PRNGKeyArray,
            latent_history: tuple[LatentT],
            condition: ConditionT,
            parameters: ParametersT,
        ) -> LatentT:
            loc_x, scale_x = self.loc_scale(latent_history, condition, parameters)
            eps = jrandom.normal(key)
            next_x = loc_x + eps * scale_x
            return self.latent_t.unravel(next_x)

        def log_prob(
            latent_history: tuple[LatentT],
            latent: LatentT,
            condition: ConditionT,
            parameters: ParametersT,
        ) -> Scalar:
            loc_x, scale_x = self.loc_scale(latent_history, condition, parameters)
            x = latent.ravel()
            lp = jstats.norm.logpdf(x, loc=loc_x, scale=scale_x)
            return jnp.sum(lp)

        super().__init__(
            order=1,
            sample=sample,
            log_prob=log_prob,
        )


class EmissionSample[
    LatentHistoryT,
    ParametersT,
    ConditionT,
    ObservationHistoryT,
    ObservationT,
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        latent: LatentHistoryT,
        observation_history: ObservationHistoryT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ObservationT: ...


class EmissionLogProb[
    LatentHistoryT,
    ConditionT,
    ObservationHistoryT,
    ObservationT,
    ParametersT,
](typing.Protocol):
    def __call__(
        self,
        latent: LatentHistoryT,
        observation: ObservationT,
        observation_history: ObservationHistoryT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...


@dataclass(frozen=True)
class Emission[
    LatentHistoryT,
    ConditionT,
    ObservationT,
    ParametersT: seqjtyping.Parameters,
    ObservationHistoryT = tuple[()],
]:
    """Emission must define density + sampling for observations at each time t.

    The emission can depend on a history of latents and observations.
    The length of these histories is determined by the dependency structure
    of the emission.
    """

    sample: EmissionSample[
        LatentHistoryT, ParametersT, ConditionT, ObservationHistoryT, ObservationT
    ]
    log_prob: EmissionLogProb[
        LatentHistoryT, ConditionT, ObservationHistoryT, ObservationT, ParametersT
    ]
    order: int
    observation_dependency: int = 0


class SequentialModel[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    # More complex dependency structure typing is handled here
    PriorLatentT: tuple[seqjtyping.Latent, ...] = tuple[LatentT],  # type: ignore
    PriorConditionT: tuple[seqjtyping.Condition, ...] = tuple[ConditionT],  # type: ignore
    TransitionLatentHistoryT: tuple[seqjtyping.Latent, ...] = tuple[LatentT],  # type: ignore
    EmissionLatentHistoryT: tuple[seqjtyping.Latent, ...] = tuple[LatentT],  # type: ignore
    ObservationHistoryT: tuple[seqjtyping.Observation, ...] = tuple[()],  # type: ignore
]:
    latent_cls: type[LatentT]
    observation_cls: type[ObservationT]
    parameter_cls: type[ParametersT]
    condition_cls: type[ConditionT]

    prior: Prior[PriorLatentT, PriorConditionT, ParametersT]
    transition: Transition[TransitionLatentHistoryT, LatentT, ConditionT, ParametersT]
    emission: Emission[
        EmissionLatentHistoryT,
        ConditionT,
        ObservationT,
        ParametersT,
        ObservationHistoryT,
    ]

    def latent_view_for_emission[ParticleHistoryT: tuple[seqjtyping.Latent, ...]](
        self,
        latent_history: ParticleHistoryT,
    ) -> EmissionLatentHistoryT:
        """Convert full latent history to emission latent history type."""
        return typing.cast(
            EmissionLatentHistoryT, latent_history[-self.emission.order :]
        )

    def latent_view_for_transition[ParticleHistoryT: tuple[seqjtyping.Latent, ...]](
        self,
        latent_history: ParticleHistoryT,
    ) -> TransitionLatentHistoryT:
        """Convert full latent history to transition latent history type."""
        return typing.cast(
            TransitionLatentHistoryT, latent_history[-self.transition.order :]
        )

    def add_observation_history(
        self, current_history: ObservationHistoryT, new_observation: ObservationT
    ) -> ObservationHistoryT:
        """Add new observation to history, maintaining correct length."""
        emission_history = (*current_history, new_observation)
        return typing.cast(
            ObservationHistoryT,
            tuple(
                emission_history[
                    len(emission_history) - self.emission.observation_dependency :
                ]
            ),
        )

    def add_latent_history[ParticleHistoryT: tuple[seqjtyping.Latent, ...]](
        self, current_history: ParticleHistoryT, new_latent: LatentT
    ) -> ParticleHistoryT:
        """Add new latent to history, maintaining correct length."""
        latent_history = (*current_history, new_latent)
        return typing.cast(
            ParticleHistoryT,
            tuple(latent_history[len(latent_history) - self.transition.order :]),
        )


class SequentialModelBase[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](ABC):
    """Method-based sequential model base class."""

    latent_cls: type[LatentT]
    observation_cls: type[ObservationT]
    parameter_cls: type[ParametersT]
    condition_cls: type[ConditionT]

    prior_order: int
    transition_order: int
    emission_order: int
    observation_dependency: int

    make_latent_context: typing.Callable[..., LatentContextType[LatentT]]
    make_observation_context: typing.Callable[..., ObservationContextType[ObservationT]]
    make_condition_context: typing.Callable[..., ConditionContextType[ConditionT]]

    def __init__(self):
        if self.prior_order < max(self.transition_order, self.emission_order):
            raise ValueError(
                "prior_order must be >= max(transition_order, emission_order), got "
                f"prior_order={self.prior_order} transition_order={self.transition_order} "
                f"emission_order={self.emission_order}"
            )
        self.make_latent_context = partial(LatentContext, max_length=self.prior_order)
        self.make_observation_context = partial(
            ObservationContext, max_length=self.observation_dependency
        )
        self.make_condition_context = partial(
            ConditionContext, max_length=self.prior_order
        )



    @property
    def prior(self):
        return SimpleNamespace(order=self.prior_order)

    @property
    def transition(self):
        return SimpleNamespace(order=self.transition_order)

    @property
    def emission(self):
        return SimpleNamespace(
            order=self.emission_order,
            observation_dependency=self.observation_dependency,
        )

    def prior_sample(
        self,
        key: PRNGKeyArray,
        conditions: ConditionContextType[ConditionT],
        parameters: ParametersT,
    ) -> LatentContextType[LatentT]:
        raise NotImplementedError

    def prior_log_prob(
        self,
        latent: LatentContextType[LatentT],
        conditions: ConditionContextType[ConditionT],
        parameters: ParametersT,
    ) -> Scalar:
        raise NotImplementedError

    def transition_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> LatentT:
        raise NotImplementedError

    def transition_log_prob(
        self,
        latent_history: LatentContext[LatentT],
        latent: LatentT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar:
        raise NotImplementedError

    def emission_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT],
        observation_history: ObservationContext[ObservationT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ObservationT:
        raise NotImplementedError

    def emission_log_prob(
        self,
        latent_history: LatentContext[LatentT],
        observation: ObservationT,
        observation_history: ObservationContext[ObservationT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar:
        raise NotImplementedError



AllSequentialModels = SequentialModelBase


class BayesianSequentialModel[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](ABC):
    inference_parameter_cls: type[InferenceParametersT]
    target: SequentialModelBase[LatentT, ObservationT, ConditionT, ParametersT]
    hyperparameters: HyperParametersT | None
    convert_to_model_parameters: Callable[[InferenceParametersT], ParametersT] = staticmethod(
        lambda x: typing.cast(ParametersT, x)
    )

    def __init__(self, hyperparameters: HyperParametersT | None = None):
        self.hyperparameters = hyperparameters

    @abstractmethod
    def parameter_prior(self) -> ParameterPrior[InferenceParametersT, HyperParametersT]:
        """Return the parameter prior object for this Bayesian model specialization."""

    def _bound_hyperparameters(self) -> HyperParametersT:
        return typing.cast(HyperParametersT, self.hyperparameters)

    def sample_inference_parameters(self, key: PRNGKeyArray) -> InferenceParametersT:
        return self.parameter_prior().sample(key, self._bound_hyperparameters())

    def log_prob_inference_parameters(self, parameters: InferenceParametersT) -> Scalar:
        return self.parameter_prior().log_prob(
            parameters, self._bound_hyperparameters()
        )
