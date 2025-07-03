"""Multidimensional linear Gaussian state space model."""

from dataclasses import field
from typing import ClassVar

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, PRNGKeyArray, Scalar

from seqjax.model.base import Emission, Prior, SequentialModel, Transition
from seqjax.model.typing import Condition, Observation, Parameters, Particle


class VectorState(Particle):
    """Multivariate latent state."""

    x: Array


class VectorObservation(Observation):
    """Vector-valued observation."""

    y: Array


class LGSSMParameters(Parameters):
    """Parameters of a linear Gaussian state space model."""

    transition_matrix: Array = field(default_factory=lambda: jnp.eye(2))
    transition_noise_scale: Array = field(default_factory=lambda: jnp.ones(2))
    emission_matrix: Array = field(default_factory=lambda: jnp.eye(2))
    emission_noise_scale: Array = field(default_factory=lambda: jnp.ones(2))


class GaussianPrior(Prior[VectorState, Condition, LGSSMParameters]):
    """Gaussian prior over the initial state."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[Condition],
        parameters: LGSSMParameters,
    ) -> tuple[VectorState]:
        mean = jnp.zeros_like(parameters.transition_noise_scale)
        scale = parameters.transition_noise_scale
        x0 = mean + scale * jrandom.normal(key, shape=scale.shape)
        return (VectorState(x=x0),)

    @staticmethod
    def log_prob(
        particle: tuple[VectorState],
        conditions: tuple[Condition],
        parameters: LGSSMParameters,
    ) -> Scalar:
        scale = parameters.transition_noise_scale
        logp = jstats.norm.logpdf(particle[0].x, loc=0.0, scale=scale)
        return logp.sum()


class GaussianTransition(Transition[VectorState, Condition, LGSSMParameters]):
    """Linear Gaussian state transition."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[VectorState],
        condition: Condition,
        parameters: LGSSMParameters,
    ) -> VectorState:
        (last_state,) = particle_history
        mean = parameters.transition_matrix @ last_state.x
        noise = parameters.transition_noise_scale * jrandom.normal(
            key, shape=parameters.transition_noise_scale.shape
        )
        return VectorState(x=mean + noise)

    @staticmethod
    def log_prob(
        particle_history: tuple[VectorState],
        particle: VectorState,
        condition: Condition,
        parameters: LGSSMParameters,
    ) -> Scalar:
        (last_state,) = particle_history
        mean = parameters.transition_matrix @ last_state.x
        logp = jstats.norm.logpdf(
            particle.x, loc=mean, scale=parameters.transition_noise_scale
        )
        return logp.sum()


class GaussianEmission(Emission[VectorState, VectorObservation, Condition, LGSSMParameters]):
    """Gaussian emission from the latent state."""

    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[VectorState],
        observation_history: tuple[()],
        condition: Condition,
        parameters: LGSSMParameters,
    ) -> VectorObservation:
        (state,) = particle
        mean = parameters.emission_matrix @ state.x
        noise = parameters.emission_noise_scale * jrandom.normal(
            key, shape=parameters.emission_noise_scale.shape
        )
        return VectorObservation(y=mean + noise)

    @staticmethod
    def log_prob(
        particle: tuple[VectorState],
        observation_history: tuple[()],
        observation: VectorObservation,
        condition: Condition,
        parameters: LGSSMParameters,
    ) -> Scalar:
        (state,) = particle
        mean = parameters.emission_matrix @ state.x
        logp = jstats.norm.logpdf(
            observation.y, loc=mean, scale=parameters.emission_noise_scale
        )
        return logp.sum()


class LinearGaussianSSM(
    SequentialModel[VectorState, VectorObservation, Condition, LGSSMParameters]
):
    particle_type = VectorState
    prior = GaussianPrior()
    transition = GaussianTransition()
    emission = GaussianEmission()
