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
import typing
from dataclasses import dataclass

from jaxtyping import PRNGKeyArray, Scalar

import seqjax.model.typing as seqjtyping


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
    """Prior must define density + sampling up to the start state t=0.
    As such it receives conditions up to t=0, length corresponding to order.

    This could be over a number of latents if the dependency structure of the model requires.

    For example if the transition is (x_t1, x_t2) -> x_t3 then the prior must produce 2 latents (x_t-1, x_t0).
    Even if the transition is (x_t1,) -> x_t2, if emission is (x_t1, x_t2) -> y_t2 then
    the prior must specify p(x_t-1, x_t0), so that p(y_t0 | x_t-1, x_t0) can be evaluated.
    """

    order: int
    sample: TransitionSample[LatentHistoryT, ConditionT, ParametersT, LatentT]
    log_prob: TransitionLogProb[LatentHistoryT, ConditionT, ParametersT, LatentT]


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

    reference_emission: ObservationHistoryT

    def latent_view_for_emission(
        self,
        latent_history: PriorLatentT,
    ) -> EmissionLatentHistoryT:
        """Convert full latent history to emission latent history type."""
        return typing.cast(
            EmissionLatentHistoryT, latent_history[-self.emission.order :]
        )

    def latent_view_for_transition(
        self,
        latent_history: PriorLatentT,
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

    def add_latent_history(
        self, current_history: PriorLatentT, new_latent: LatentT
    ) -> PriorLatentT:
        """Add new latent to history, maintaining correct length."""
        latent_history = (*current_history, new_latent)
        return typing.cast(
            PriorLatentT,
            tuple(latent_history[len(latent_history) - self.transition.order :]),
        )


# Specify a version of SequentialModel with 2nd order transitions
class SequentialModel_TO2_EO2[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    SequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        tuple[LatentT, LatentT],
        tuple[ConditionT, ConditionT],
        tuple[LatentT, LatentT],
        tuple[LatentT, LatentT],
        tuple[()],
    ]
): ...


AllSequentialModels = SequentialModel | SequentialModel_TO2_EO2


class BayesianSequentialModel[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
    # More complex dependency structure typing is handled here
    PriorLatentT: tuple[seqjtyping.Latent, ...] = tuple[LatentT],  # type: ignore
    PriorConditionT: tuple[seqjtyping.Condition, ...] = tuple[ConditionT],  # type: ignore
    TransitionLatentHistoryT: tuple[seqjtyping.Latent, ...] = tuple[LatentT],  # type: ignore
    EmissionLatentHistoryT: tuple[seqjtyping.Latent, ...] = tuple[LatentT],  # type: ignore
    ObservationHistoryT: tuple[seqjtyping.Observation, ...] = tuple[()],  # type: ignore
]:
    inference_parameter_cls: type[InferenceParametersT]
    target: SequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        PriorLatentT,
        PriorConditionT,
        TransitionLatentHistoryT,
        EmissionLatentHistoryT,
        ObservationHistoryT,
    ]
    parameter_prior: ParameterPrior[InferenceParametersT, HyperParametersT]
    target_parameter: Callable[[InferenceParametersT], ParametersT]


class BayesianSequentialModel_TO2_EO2[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    BayesianSequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
        tuple[LatentT, LatentT],
        tuple[ConditionT, ConditionT],
        tuple[LatentT, LatentT],
        tuple[LatentT, LatentT],
        tuple[()],
    ]
): ...
