"""AR(1) example model implementations."""

from dataclasses import field
from typing import ClassVar

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    Condition,
    Emission,
    HyperParameters,
    Observation,
    ParameterPrior,
    Parameters,
    Particle,
    Prior,
    Target,
    Transition,
)


class LatentValue(Particle):
    """Latent AR state."""

    x: Scalar


class ARParameters(Parameters):
    """Parameters of the AR(1) model."""

    ar: Scalar = field(default_factory=lambda: jnp.array(0.5))
    observation_std: Scalar = field(default_factory=lambda: jnp.array(1.0))
    transition_std: Scalar = field(default_factory=lambda: jnp.array(0.5))


class NoisyEmission(Observation):
    """Observation wrapping a scalar value."""

    y: Scalar


class HalfCauchyStds(ParameterPrior[ARParameters, HyperParameters]):
    """Half-Cauchy priors for the model standard deviations."""

    @staticmethod
    def sample(key: PRNGKeyArray, _hyperparameters: HyperParameters) -> ARParameters:
        ar_key, o_std_key, t_std_key = jrandom.split(key, 3)
        return ARParameters(
            ar=jrandom.uniform(ar_key, minval=-1, maxval=1),
            observation_std=jnp.abs(jrandom.cauchy(o_std_key)),
            transition_std=jnp.abs(jrandom.cauchy(t_std_key)),
        )

    @staticmethod
    def log_p(parameteters: ARParameters, _hyperparameters: HyperParameters) -> Scalar:
        """Evaluate the log-density of ``parameteters`` under the prior."""
        log_p_theta = jstats.uniform.logpdf(parameteters.ar, loc=-1.0, scale=2.0)
        log_2 = jnp.log(jnp.array(2.0))
        log_p_theta += jstats.cauchy.logpdf(parameteters.observation_std) + log_2
        log_p_theta += jstats.cauchy.logpdf(parameteters.transition_std) + log_2
        return log_p_theta


class InitialValue(Prior[LatentValue, Condition, ARParameters]):
    """Gaussian prior over the initial latent state."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        _conditions: tuple[Condition],
        parameters: ARParameters,
    ) -> tuple[LatentValue]:
        """Sample the initial latent value."""
        x0 = parameters.transition_std * jrandom.normal(
            key,
        )
        return (LatentValue(x=x0),)

    @staticmethod
    def log_p(
        particle: tuple[LatentValue],
        _conditions: tuple[Condition],
        parameters: ARParameters,
    ) -> Scalar:
        """Evaluate the prior log-density."""
        return jstats.norm.logpdf(particle[0].x, scale=parameters.transition_std)


class ARRandomWalk(Transition[LatentValue, Condition, ARParameters]):
    """Gaussian AR(1) transition."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[LatentValue],
        _condition: Condition,
        parameters: ARParameters,
    ) -> LatentValue:
        """Sample the next latent state."""
        last_particle, = particle_history
        next_x = (
            parameters.ar * last_particle.x
            + jrandom.normal(key) * parameters.transition_std
        )
        return LatentValue(x=next_x)

    @staticmethod
    def log_p(
        particle_history: tuple[LatentValue],
        particle: LatentValue,
        _condition: Condition,
        parameters: ARParameters,
    ) -> Scalar:
        """Return the transition log-density."""
        last_particle, = particle_history
        return jstats.norm.logpdf(
            particle.x,
            loc=parameters.ar * last_particle.x,
            scale=parameters.transition_std,
        )


class AREmission(Emission[LatentValue, NoisyEmission, Condition,  ARParameters]):
    """Normal emission from the latent state."""

    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[LatentValue],
        _observation_history: tuple[()],
        _condition: Condition,
        parameters: ARParameters,
    ) -> NoisyEmission:
        """Sample an observation."""
        current_particle, = particle
        y = current_particle.x + jrandom.normal(key) * parameters.observation_std
        return NoisyEmission(y=y)

    @staticmethod
    def log_p(
        particle: tuple[LatentValue],
        _observation_history: tuple[()],
        observation: NoisyEmission,
        _condition: Condition,
        parameters: ARParameters,
    ) -> Scalar:
        """Return the emission log-density."""
        current_particle, = particle
        return jstats.norm.logpdf(
            observation.y,
            loc=current_particle.x,
            scale=parameters.observation_std,
        )


class AR1Target(Target[LatentValue, NoisyEmission, Condition,  ARParameters]):
    particle_type = LatentValue
    prior = InitialValue()
    transition= ARRandomWalk()
    emission=AREmission()

