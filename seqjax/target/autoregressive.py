from dataclasses import dataclass, field

from seqjax.target.base import (
    Target,
    Particle,
    Observation,
    Condition,
    Parameters,
    Transition,
    Prior,
    Emission,
)

from jaxtyping import Scalar, PRNGKeyArray
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.random as jrandom


class LatentValue(Particle):
    x: Scalar


class ARParameters(Parameters):
    ar: Scalar = field(default_factory=lambda: jnp.array(0.5))
    observation_std: Scalar = field(default_factory=lambda: jnp.array(1.0))
    transition_std: Scalar = field(default_factory=lambda: jnp.array(0.5))


class NoisyObservation(Observation):
    y: Scalar


class InitialValue(Prior[LatentValue, ARParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, parameters: ARParameters) -> LatentValue:
        x0 = parameters.transition_std * jrandom.normal(
            key,
        )
        return LatentValue(x=x0)

    @staticmethod
    def log_p(particle: LatentValue, parameters: ARParameters) -> Scalar:
        return jstats.norm.logpdf(particle.x, scale=parameters.transition_std)


class ARRandomWalk(Transition[LatentValue, Condition, ARParameters]):
    @staticmethod
    def sample(key, particle, condition, parameters):
        next_x = (
            parameters.ar * particle.x + jrandom.normal(key) * parameters.transition_std
        )
        return LatentValue(x=next_x)

    @staticmethod
    def log_p(particle, next_particle, condition, parameters):
        return jstats.norm.logpdf(
            next_particle.x,
            loc=parameters.ar * particle.x,
            scale=parameters.transition_std,
        )


class AREmission(Emission[LatentValue, Condition, NoisyObservation, ARParameters]):
    @staticmethod
    def sample(key, particle, condition, parameters):
        y = particle.x + jrandom.normal(key) * parameters.observation_std
        return NoisyObservation(y=y)

    @staticmethod
    def log_p(particle, observation, condition, parameters):
        return jstats.norm.logpdf(
            observation.y,
            loc=particle.x,
            scale=parameters.observation_std,
        )
