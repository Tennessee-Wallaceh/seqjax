"""Core interfaces for state-space model components using Equinox modules.

This module defines the structural interfaces for probabilistic models using
Equinox's `eqx.Module`. Components include `Prior`, `Transition`, `Emission`,
and the top-level container `Target`.

Notes:
- `eqx.Module` is used both as a PyTree-compatible dataclass and as a container
  for grouping static methods that define pure functional behavior.
- Components are structured as "struct-of-arrays" to support efficient batched
  inference and JAX transformations.
- `order` refers to the number of past latent states required by a transition
  or emission component (e.g. in a higher-order Markov process).
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Generic

import equinox as eqx
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.typing import (
    ConditionType,
    EnforceInterface,
    HyperParametersType,
    ObservationType,
    ParametersType,
    ParticleType,
)


class ParameterPrior(eqx.Module, Generic[ParametersType, HyperParametersType]):
    """Represents a prior distribution over model parameters in a Bayesian model.

    This interface allows specification of a log-density function (and optionally
    a sampler) for parameters given hyperparameters. It is designed for use in
    hierarchical or fully Bayesian models, where priors over parameters are treated
    as first-class components.

    Example usage includes defining priors for neural network weights, transition
    matrices, or noise scales.

    Note:
        The `log_p` method must be implemented to return log p(parameters | hyperparameters).

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
    """Defines the initial state distribution p(x_{-k:0} | condition, parameters).

    The prior must define the density and provide sampling up to the start time t = 0.
    It receives conditions up to t = 0, with a length determined by the model order.

    This could involve multiple particles if required by the dependency structure.
    For example, if the transition is (x_{t-2}, x_{t-1}) → x_t, then the prior must produce
    two initial particles (x_{-1}, x_0).

    Even if the transition is (x_{t-1},) → x_t, but the emission is (x_{t-1}, x_t) → y_t,
    the prior must define p(x_{-1}, x_0), so that p(y_0 | x_{-1}, x_0) can be evaluated.
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
    """Defines the state transition distribution p(x_t | x_{t-k:t-1}, condition, parameters).

    The transition operates over a fixed-order Markov history of particles.
    The class must implement sampling and log-density evaluation given:
    - A particle history of length equal to `order`,
    - External conditions (e.g., exogenous inputs),
    - Parameters of the model.

    Attributes:
        order: The number of previous particles the transition depends on (k).
    """

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
    """Defines the emission distribution p(y_t | x_{t-k:t}, y_{t-l:t-1}, condition, parameters).

    The emission depends on a history of particles and past observations, as well
    as conditions and parameters. The temporal dependencies are explicitly encoded
    by `order` (particle history length) and `observation_dependency` (observation history length).

    Attributes:
        order: Number of past particles the emission depends on (k).
        observation_dependency: Number of past observations it depends on (l).

    """

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
    """A complete probabilistic model p(x_{1:T}, y_{1:T} | condition, parameters).

    A `Target` defines the full generative model through its three components:

    - `prior`: Specifies the initial state distribution p(x_{-k:0} | condition, parameters).
    - `transition`: Defines the Markovian dynamics p(x_t | x_{t-k:t-1}, condition, parameters).
    - `emission`: Models observations p(y_t | x_{t-k:t}, y_{t-l:t-1}, condition, parameters).

    This structure supports arbitrary Markov order and observation history dependencies
    via the attributes `order` and `observation_dependency` on the component classes.
    """

    prior: Prior[ParticleType, ConditionType, ParametersType]
    transition: Transition[ParticleType, ConditionType, ParametersType]
    emission: Emission[ParticleType, ObservationType, ConditionType, ParametersType]

    particle_class: Callable[[], ParticleType]
    observation_class: Callable[[], ObservationType]
