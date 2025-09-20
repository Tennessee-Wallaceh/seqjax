"""Multidimensional linear Gaussian state space model."""

from collections import OrderedDict
from dataclasses import field
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, PRNGKeyArray, Scalar

from seqjax.model.base import Emission, Prior, SequentialModel, Transition
from seqjax.model.typing import Observation, Parameters, Particle, NoCondition


class VectorState(Particle):
    """Multivariate latent state."""

    x: Array

    _shape_template: ClassVar = OrderedDict(
        x=jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
    )


class VectorObservation(Observation):
    """Vector-valued observation."""

    y: Array

    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
    )


class LGSSMParameters(Parameters):
    """Parameters of a linear Gaussian state space model."""

    transition_matrix: Array = field(default_factory=lambda: jnp.eye(2))
    transition_noise_scale: Array = field(default_factory=lambda: jnp.ones(2))
    emission_matrix: Array = field(default_factory=lambda: jnp.eye(2))
    emission_noise_scale: Array = field(default_factory=lambda: jnp.ones(2))

    _shape_template: ClassVar = OrderedDict(
        transition_matrix=jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
        transition_noise_scale=jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
        emission_matrix=jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
        emission_noise_scale=jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
    )


class GaussianPrior(Prior[tuple[VectorState], tuple[()], LGSSMParameters]):
    """Gaussian prior over the initial state."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[()],
        parameters: LGSSMParameters,
    ) -> tuple[VectorState]:
        mean = jnp.zeros_like(parameters.transition_noise_scale)
        scale = parameters.transition_noise_scale
        x0 = mean + scale * jrandom.normal(key, shape=scale.shape)
        return (VectorState(x=x0),)

    @staticmethod
    def log_prob(
        particle: tuple[VectorState],
        conditions: tuple[()],
        parameters: LGSSMParameters,
    ) -> Scalar:
        scale = parameters.transition_noise_scale
        logp = jstats.norm.logpdf(particle[0].x, loc=0.0, scale=scale)
        return logp.sum()


class GaussianTransition(
    Transition[VectorState, tuple[VectorState], NoCondition, LGSSMParameters]
):
    """Linear Gaussian state transition."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[VectorState],
        condition: NoCondition,
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
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> Scalar:
        (last_state,) = particle_history
        mean = parameters.transition_matrix @ last_state.x
        logp = jstats.norm.logpdf(
            particle.x, loc=mean, scale=parameters.transition_noise_scale
        )
        return logp.sum()


class GaussianEmission(
    Emission[
        tuple[VectorState],
        VectorObservation,
        tuple[()],
        NoCondition,
        LGSSMParameters,
    ]
):
    """Gaussian emission from the latent state."""

    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[VectorState],
        observation_history: tuple[()],
        condition: NoCondition,
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
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> Scalar:
        (state,) = particle
        mean = parameters.emission_matrix @ state.x
        logp = jstats.norm.logpdf(
            observation.y, loc=mean, scale=parameters.emission_noise_scale
        )
        return logp.sum()


class LinearGaussianSSM(
    SequentialModel[
        VectorState,
        tuple[VectorState],
        tuple[VectorState],
        tuple[VectorState],
        VectorObservation,
        tuple[()],
        tuple[()],
        NoCondition,
        LGSSMParameters,
    ]
):
    particle_cls = VectorState
    observation_cls = VectorObservation
    parameter_cls = LGSSMParameters
    prior = GaussianPrior()
    transition = GaussianTransition()
    emission = GaussianEmission()
