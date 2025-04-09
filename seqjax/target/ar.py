from dataclasses import dataclass, field

from seqjax.target.base import (
    Target,
    Particle,
    Observation,
    Condition,
    Parameters,
    HyperParameters,
    Transition,
    ParameterPrior,
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
    

class AR1Target(Target[LatentValue, Condition, NoisyObservation, ARParameters]):
    particle_type = LatentValue
    prior = InitialValue
    transition= ARRandomWalk
    emission=AREmission