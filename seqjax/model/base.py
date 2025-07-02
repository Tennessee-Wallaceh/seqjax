from typing import Generic
from abc import abstractmethod
import equinox as eqx
from jaxtyping import PRNGKeyArray, Scalar
from seqjax.model.typing import (
    EnforceInterface,
    ParametersType,
    ParticleType,
    ObservationType,
    ConditionType,
    HyperParametersType,
)

"""
Use equinox for eqx.Module as pytree + dataclass, providing struct of array definitions.
Also use eqx.Module to group pure functions expressed as static methods.

Notes:
- order refers to the length of the history of the latent.

"""
# Condition and Parameters are seperated, the idea being parameters are static across time
# condition is varying.

# use Target class to group together a number of pure functions (staticmethod), that operate on the same
# Particle, Emission etc.
# Prior, Transition, Emission give additional levels of grouping, since in practice these will always be paired.


class ParameterPrior(eqx.Module, Generic[ParametersType, HyperParametersType]):
    """
    Parameter prior specified as utility for specifying Bayesian models.
    """

    # @staticmethod
    # @abstractmethod
    # def sample(
    #     key: PRNGKeyArray, hyperparameters: ParametersType
    # ) -> ParametersType: ...

    @staticmethod
    @abstractmethod
    def log_p(
        parameters: ParametersType, hyperparameters: HyperParametersType
    ) -> Scalar: ...


class Prior(
    eqx.Module, Generic[ParticleType, ConditionType, ParametersType], EnforceInterface
):
    """
    Prior must define density + sampling up to the start state t=0.
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
    def log_p(
        particle: tuple[ParticleType, ...],
        conditions: tuple[ConditionType, ...],
        parameters: ParametersType,
    ) -> Scalar: ...


class Transition(
    eqx.Module, Generic[ParticleType, ConditionType, ParametersType], EnforceInterface
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
    def log_p(
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
    def log_p(
        particle: tuple[ParticleType, ...],
        observation_history: tuple[ObservationType, ...],
        observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar: ...


class Target(Generic[ParticleType, ObservationType, ConditionType, ParametersType]):
    prior: Prior[ParticleType, ConditionType, ParametersType]
    transition: Transition[ParticleType, ConditionType, ParametersType]
    emission: Emission[ParticleType, ObservationType, ConditionType, ParametersType]
