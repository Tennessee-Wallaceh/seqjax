"""Simple Poisson state space model."""

from dataclasses import field
from typing import ClassVar

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from .base import Emission, Prior, SequentialModel, Transition
from .typing import Condition, Observation, Parameters, Particle


class LogRate(Particle):
    """Latent log-intensity."""

    log_rate: Scalar


class CountObservation(Observation):
    """Observed counts."""

    count: Scalar


class PoissonSSMParameters(Parameters):
    """Model parameters."""

    ar_coeff: Scalar = field(default_factory=lambda: jnp.array(0.9))
    transition_std: Scalar = field(default_factory=lambda: jnp.array(0.1))


class GaussianPrior(Prior[LogRate, Condition, PoissonSSMParameters]):
    """Gaussian prior over the initial log-rate."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[Condition],
        parameters: PoissonSSMParameters,
    ) -> tuple[LogRate]:
        init = parameters.transition_std * jrandom.normal(key)
        return (LogRate(log_rate=init),)

    @staticmethod
    def log_prob(
        particle: tuple[LogRate],
        conditions: tuple[Condition],
        parameters: PoissonSSMParameters,
    ) -> Scalar:
        return jstats.norm.logpdf(
            particle[0].log_rate, scale=parameters.transition_std
        )


class GaussianRW(Transition[LogRate, Condition, PoissonSSMParameters]):
    """Gaussian AR(1) transition on the log-rate."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[LogRate],
        condition: Condition,
        parameters: PoissonSSMParameters,
    ) -> LogRate:
        (prev,) = particle_history
        loc = parameters.ar_coeff * prev.log_rate
        noise = parameters.transition_std * jrandom.normal(key)
        return LogRate(log_rate=loc + noise)

    @staticmethod
    def log_prob(
        particle_history: tuple[LogRate],
        particle: LogRate,
        condition: Condition,
        parameters: PoissonSSMParameters,
    ) -> Scalar:
        (prev,) = particle_history
        loc = parameters.ar_coeff * prev.log_rate
        return jstats.norm.logpdf(
            particle.log_rate, loc=loc, scale=parameters.transition_std
        )


class PoissonEmission(
    Emission[LogRate, CountObservation, Condition, PoissonSSMParameters]
):
    """Poisson emission with rate ``exp(log_rate)``."""

    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[LogRate],
        observation_history: tuple[()],
        condition: Condition,
        parameters: PoissonSSMParameters,
    ) -> CountObservation:
        (current,) = particle
        rate = jnp.exp(current.log_rate)
        return CountObservation(count=jrandom.poisson(key, rate))

    @staticmethod
    def log_prob(
        particle: tuple[LogRate],
        observation_history: tuple[()],
        observation: CountObservation,
        condition: Condition,
        parameters: PoissonSSMParameters,
    ) -> Scalar:
        (current,) = particle
        rate = jnp.exp(current.log_rate)
        return jstats.poisson.logpmf(observation.count, rate)


class PoissonSSM(
    SequentialModel[LogRate, CountObservation, Condition, PoissonSSMParameters]
):
    particle_type = LogRate
    prior = GaussianPrior()
    transition = GaussianRW()
    emission = PoissonEmission()

