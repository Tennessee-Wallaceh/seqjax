"""Abstract base classes for sequential models.

Condition and Parameters are separated. Parameters remain static over time while
conditions vary.

Use ``SequentialModel`` to group pure functions that operate on the same
``Particle`` and ``Emission`` types. ``Prior``, ``Transition`` and ``Emission``
provide additional structure and are typically paired in use.
"""

from abc import abstractmethod
from typing import Callable

import equinox as eqx
from jaxtyping import PRNGKeyArray, Scalar

import seqjax.model.typing as seqjtyping


class ParameterPrior[
    ParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](eqx.Module):
    """Parameter prior specified as utility for specifying Bayesian models."""

    @staticmethod
    @abstractmethod
    def log_prob(
        parameters: ParametersT,
        hyperparameters: HyperParametersT,
    ) -> Scalar: ...

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        hyperparameters: HyperParametersT,
    ) -> ParametersT: ...


class Prior[
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...] | None,
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
    seqjtyping.EnforceInterface,
):
    """Prior must define density + sampling up to the start state t=0.
    As such it receives conditions up to t=0, length corresponding to order.

    This could be over a number of particles if the dependency structure of the model requires.

    For example if the transition is (x_t1, x_t2) -> x_t3 then the prior must produce 2 particles (x_t-1, x_t0).
    Even if the transition is (x_t1,) -> x_t2, if emission is (x_t1, x_t2) -> y_t2 then
    the prior must specify p(x_t-1, x_t0), so that p(y_t0 | x_t-1, x_t0) can be evaluated.
    """

    order: eqx.AbstractClassVar[int]  # 1 + max(Transition.order - 1, Emission.order)

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        conditions: ConditionHistoryT,
        parameters: ParametersT,
    ) -> InitialParticleT: ...

    @staticmethod
    @abstractmethod
    def log_prob(
        particle: InitialParticleT,
        conditions: ConditionHistoryT,
        parameters: ParametersT,
    ) -> Scalar: ...


class Transition[
    ParticleT: seqjtyping.Particle,
    ParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ConditionT: seqjtyping.Condition | None,
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
    seqjtyping.EnforceInterface,
):
    order: eqx.AbstractClassVar[int]

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: ParticleHistoryT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ParticleT: ...

    @staticmethod
    @abstractmethod
    def log_prob(
        particle_history: ParticleHistoryT,
        particle: ParticleT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...


class Emission[
    ParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionT: seqjtyping.Condition | None,
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
    seqjtyping.EnforceInterface,
):
    order: eqx.AbstractClassVar[int]
    observation_dependency: eqx.AbstractClassVar[int]

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        particle: ParticleHistoryT,
        observation_history: ObservationHistoryT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ObservationT: ...

    @staticmethod
    @abstractmethod
    def log_prob(
        particle: ParticleHistoryT,
        observation_history: ObservationHistoryT,
        observation: ObservationT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...


class SequentialModel[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...] | None,
    ConditionT: seqjtyping.Condition | None,
    ParametersT: seqjtyping.Parameters,
]:
    particle_cls: type[ParticleT]
    observation_cls: type[ObservationT]
    parameter_cls: type[ParametersT]
    prior: Prior[InitialParticleT, ConditionHistoryT, ParametersT]
    transition: Transition[
        ParticleT, TransitionParticleHistoryT, ConditionT, ParametersT
    ]
    emission: Emission[
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionT,
        ParametersT,
    ]


class BayesianSequentialModel[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
]:
    inference_parameter_cls: type[InferenceParametersT]
    target: SequentialModel[
        ParticleT,
        InitialParticleT,
        TransitionParticleHistoryT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
        ConditionT,
        ParametersT,
    ]
    parameter_prior: ParameterPrior[InferenceParametersT, HyperParametersT]
    target_parameter: Callable[[InferenceParametersT], ParametersT]
