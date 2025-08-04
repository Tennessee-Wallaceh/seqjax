"""Logistic regression SMC sampler example.

This module shows how an annealed sequential Monte Carlo sampler can be built
using the :class:`~seqjax.model.base.SequentialModel` interfaces.

Example
-------
>>> from seqjax.model.logistic_smc import LogisticRegressionSMC
>>> smc = LogisticRegressionSMC()
"""

from dataclasses import field
from typing import ClassVar

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import Emission, Prior, SequentialModel, Transition
from seqjax.model.typing import (
    Condition,
    Observation,
    Parameters,
    Particle,
)


class LogRegParticle(Particle):
    """Latent logistic regression parameters."""

    theta: jnp.ndarray  # shape (d,)


class LogRegData(Parameters):
    """Fixed dataset used as model parameters."""

    X: jnp.ndarray  # shape (n, d)
    y: jnp.ndarray  # shape (n,)
    reference_emission: tuple[Observation] = field(default_factory=tuple)


class DummyObservation(Observation):
    """Placeholder observation carrying a scalar value."""

    dummy: Scalar = field(default_factory=lambda: jnp.array(0.0))


class AnnealCondition(Condition):
    """Annealing level."""

    beta: Scalar
    beta_prev: Scalar


class GaussianPrior(Prior[LogRegParticle, AnnealCondition, LogRegData]):
    """Wide Gaussian prior over the regression coefficients."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[AnnealCondition],
        parameters: LogRegData,
    ) -> tuple[LogRegParticle]:
        (cond,) = conditions
        _ = cond  # unused
        theta = 10.0 * jrandom.normal(key, shape=(parameters.X.shape[1],))
        return (LogRegParticle(theta=theta),)

    @staticmethod
    def log_prob(
        particle: tuple[LogRegParticle],
        conditions: tuple[AnnealCondition],
        parameters: LogRegData,
    ) -> Scalar:
        (theta,) = particle
        return jstats.norm.logpdf(theta.theta, loc=0.0, scale=10.0).sum()


class RWTransition(Transition[LogRegParticle, AnnealCondition, LogRegData]):
    """Random walk Metropolis proposal (symmetric)."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[LogRegParticle],
        condition: AnnealCondition,
        parameters: LogRegData,
    ) -> LogRegParticle:
        (state,) = particle_history
        step = 0.5 * jrandom.normal(key, shape=state.theta.shape)
        return LogRegParticle(theta=state.theta + step)

    @staticmethod
    def log_prob(
        particle_history: tuple[LogRegParticle],
        particle: LogRegParticle,
        condition: AnnealCondition,
        parameters: LogRegData,
    ) -> Scalar:
        return jnp.array(0.0)


class TemperedEmission(
    Emission[LogRegParticle, DummyObservation, AnnealCondition, LogRegData]
):
    """Incremental weight using tempering."""

    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[LogRegParticle],
        observation_history: tuple[()],
        condition: AnnealCondition,
        parameters: LogRegData,
    ) -> DummyObservation:
        return DummyObservation()

    @staticmethod
    def log_prob(
        particle: tuple[LogRegParticle],
        observation_history: tuple[()],
        observation: DummyObservation,
        condition: AnnealCondition,
        parameters: LogRegData,
    ) -> Scalar:
        (state,) = particle
        logits = parameters.X @ state.theta
        loglik = jnp.sum(parameters.y * logits - jnp.logaddexp(0.0, logits))
        return (condition.beta - condition.beta_prev) * loglik


class LogisticRegressionSMC(
    SequentialModel[LogRegParticle, DummyObservation, AnnealCondition, LogRegData]
):
    """SequentialModel wrapping the tempered logistic regression components."""

    prior = GaussianPrior()
    transition = RWTransition()
    emission = TemperedEmission()

