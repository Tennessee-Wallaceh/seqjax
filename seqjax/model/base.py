"""Abstract base classes for sequential models.

Condition and Parameters are separated. Parameters remain static over time while
conditions vary.

Use ``SequentialModel`` to group pure functions that operate on the same
``Particle`` and ``Emission`` types. ``Prior``, ``Transition`` and ``Emission``
provide additional structure and are typically paired in use.
"""

from abc import abstractmethod
from typing import Generic, Callable

import equinox as eqx
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.typing import (
    ConditionType,
    EnforceInterface,
    HyperParametersType,
    ObservationType,
    ParametersType,
    InferenceParametersType,
    ParticleType,
)


class ParameterPrior(eqx.Module, Generic[ParametersType, HyperParametersType]):
    """Parameter prior specified as utility for specifying Bayesian models."""

    @staticmethod
    @abstractmethod
    def log_prob(
        parameters: ParametersType,
        hyperparameters: HyperParametersType,
    ) -> Scalar: ...

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        hyperparameters: HyperParametersType,
    ) -> ParametersType: ...


class Prior(
    eqx.Module,
    Generic[ParticleType, ConditionType, ParametersType],
    EnforceInterface,
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
        conditions: tuple[ConditionType, ...],
        parameters: ParametersType,
    ) -> tuple[ParticleType, ...]: ...

    @staticmethod
    @abstractmethod
    def log_prob(
        particle: tuple[ParticleType, ...],
        conditions: tuple[ConditionType, ...],
        parameters: ParametersType,
    ) -> Scalar: ...


class Transition(
    eqx.Module,
    Generic[ParticleType, ConditionType, ParametersType],
    EnforceInterface,
):
    order: eqx.AbstractClassVar[int]

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[ParticleType, ...],
        condition: ConditionType,
        parameters: ParametersType,
    ) -> ParticleType: ...

    @staticmethod
    @abstractmethod
    def log_prob(
        particle_history: tuple[ParticleType, ...],
        particle: ParticleType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar: ...


class Emission(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType],
    EnforceInterface,
):
    order: eqx.AbstractClassVar[int]
    observation_dependency: eqx.AbstractClassVar[int]

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[ParticleType, ...],
        observation_history: tuple[ObservationType, ...],
        condition: ConditionType,
        parameters: ParametersType,
    ) -> ObservationType: ...

    @staticmethod
    @abstractmethod
    def log_prob(
        particle: tuple[ParticleType, ...],
        observation_history: tuple[ObservationType, ...],
        observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar: ...


class SequentialModel(
    Generic[ParticleType, ObservationType, ConditionType, ParametersType]
):
    particle_cls: type[ParticleType]
    observation_cls: type[ObservationType]
    parameter_cls: type[ParametersType]
    prior: Prior[ParticleType, ConditionType, ParametersType]
    transition: Transition[ParticleType, ConditionType, ParametersType]
    emission: Emission[ParticleType, ObservationType, ConditionType, ParametersType]


"""
There are different usage patterns to support here
- run inference for latent with fixed 
"""


class BayesianSequentialModel(
    Generic[
        ParticleType,
        ObservationType,
        ConditionType,
        ParametersType,
        InferenceParametersType,
        HyperParametersType,
    ]
):
    inference_parameter_cls: type[InferenceParametersType]
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
    parameter_prior: ParameterPrior[InferenceParametersType, HyperParametersType]
    target_parameter: Callable[[InferenceParametersType], ParametersType]
