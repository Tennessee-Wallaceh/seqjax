from dataclasses import dataclass, field
from typing import Union, ClassVar
from seqjax.model.base import (
    Target,
    Particle,
    Emission,
    Condition,
    Parameters,
    HyperParameters,
    Transition,
    ParameterPrior,
    Prior,
    Observation,
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


class NoisyEmission(Observation):
    y: Scalar


class HalfCauchyStds(ParameterPrior[ARParameters, HyperParameters]):
    @staticmethod
    def sample(key, hyperparameters):
        ar_key, o_std_key, t_std_key = jrandom.split(key, 3)
        return ARParameters(
            ar=jrandom.uniform(ar_key, minval=-1, maxval=1),
            observation_std=jnp.abs(jrandom.cauchy(o_std_key)),
            transition_std=jnp.abs(jrandom.cauchy(t_std_key)),
        )

    @staticmethod
    def log_p(parameteters, hyperparameters):
        log_p_theta = jstats.uniform.logpdf(parameteters.ar, loc=-1.0, scale=2.0)
        log_2 = jnp.log(jnp.array(2.0))
        log_p_theta += jstats.cauchy.logpdf(parameteters.observation_std) + log_2
        log_p_theta += jstats.cauchy.logpdf(parameteters.transition_std) + log_2
        return log_p_theta


class InitialValue(Prior[LatentValue, Condition, ARParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(key: PRNGKeyArray, conditions: tuple[Condition], parameters: ARParameters) -> tuple[LatentValue]:
        x0 = parameters.transition_std * jrandom.normal(
            key,
        )
        return (LatentValue(x=x0),)

    @staticmethod
    def log_p(particle: tuple[LatentValue], conditions: tuple[Condition], parameters: ARParameters) -> Scalar:
        return jstats.norm.logpdf(particle[0].x, scale=parameters.transition_std)


class ARRandomWalk(Transition[LatentValue, Condition, ARParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray, 
        particle_history: tuple[LatentValue], 
        condition: Condition, 
        parameters: ARParameters,
    ) -> LatentValue:
        last_particle, = particle_history
        next_x = (
            parameters.ar * last_particle.x + jrandom.normal(key) * parameters.transition_std
        )
        return LatentValue(x=next_x)

    @staticmethod
    def log_p(
        particle_history: tuple[LatentValue], 
        particle: LatentValue, 
        condition: Condition, 
        parameters: ARParameters
    ) -> Scalar:
        last_particle, = particle_history
        return jstats.norm.logpdf(
            particle.x,
            loc=parameters.ar * last_particle.x,
            scale=parameters.transition_std,
        )


class AREmission(Emission[LatentValue, NoisyEmission, Condition,  ARParameters]):
    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray, 
        particle: tuple[LatentValue], 
        observation_history: tuple[()],
        condition: Condition, 
        parameters: ARParameters,
    ) -> NoisyEmission:
        current_particle, = particle
        y = current_particle.x + jrandom.normal(key) * parameters.observation_std
        return NoisyEmission(y=y)

    @staticmethod
    def log_p(
        particle: tuple[LatentValue], 
        observation_history: tuple[()],
        observation: NoisyEmission,
        condition: Condition, 
        parameters: ARParameters,
    ) -> Scalar:
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