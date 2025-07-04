"""Simple stochastic SIR model."""

from dataclasses import field
from typing import ClassVar

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import Emission, Prior, SequentialModel, Transition
from seqjax.model.typing import Condition, Observation, Parameters, Particle


class SIRState(Particle):
    """Susceptible--infected--recovered counts."""

    s: Scalar
    i: Scalar
    r: Scalar


class InfectionObservation(Observation):
    """Number of newly infected individuals."""

    new_cases: Scalar


class SIRParameters(Parameters):
    """Infection and recovery rates."""

    infection_rate: Scalar = field(default_factory=lambda: jnp.array(0.3))
    recovery_rate: Scalar = field(default_factory=lambda: jnp.array(0.1))
    population: Scalar = field(default_factory=lambda: jnp.array(1000.0))
    reference_emission: tuple[InfectionObservation] = field(default_factory=tuple)


class SIRPrior(Prior[SIRState, Condition, SIRParameters]):
    """Deterministic initial counts."""

    order: ClassVar[int] = 2

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[Condition, Condition],
        parameters: SIRParameters,
    ) -> tuple[SIRState, SIRState]:
        s0 = parameters.population - 1
        state = SIRState(s=s0, i=jnp.array(1.0), r=jnp.array(0.0))
        return (state, state)

    @staticmethod
    def log_prob(
        particle: tuple[SIRState, SIRState],
        conditions: tuple[Condition, Condition],
        parameters: SIRParameters,
    ) -> Scalar:
        s0 = parameters.population - 1
        cond = (
            (particle[0].s == s0)
            & (particle[0].i == 1.0)
            & (particle[0].r == 0.0)
            & (particle[1].s == s0)
            & (particle[1].i == 1.0)
            & (particle[1].r == 0.0)
        )
        return jnp.where(cond, jnp.array(0.0), -jnp.inf)


class SIRTransition(Transition[SIRState, Condition, SIRParameters]):
    """Stochastic discrete-time SIR transition."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[SIRState],
        condition: Condition,
        parameters: SIRParameters,
    ) -> SIRState:
        (state,) = particle_history
        key_inf, key_rec = jrandom.split(key)
        lam_inf = parameters.infection_rate * state.s * state.i / parameters.population
        new_inf = jrandom.poisson(key_inf, lam_inf)
        new_inf = jnp.minimum(new_inf, state.s)
        i_temp = state.i + new_inf
        lam_rec = parameters.recovery_rate * i_temp
        new_rec = jrandom.poisson(key_rec, lam_rec)
        new_rec = jnp.minimum(new_rec, i_temp)
        s = state.s - new_inf
        i = i_temp - new_rec
        r = state.r + new_rec
        return SIRState(s=s, i=i, r=r)

    @staticmethod
    def log_prob(
        particle_history: tuple[SIRState],
        particle: SIRState,
        condition: Condition,
        parameters: SIRParameters,
    ) -> Scalar:
        (state,) = particle_history
        new_inf = state.s - particle.s
        i_temp = state.i + new_inf
        new_rec = particle.r - state.r
        lam_inf = parameters.infection_rate * state.s * state.i / parameters.population
        lam_rec = parameters.recovery_rate * i_temp
        log_p_inf = jstats.poisson.logpmf(new_inf, lam_inf)
        log_p_rec = jstats.poisson.logpmf(new_rec, lam_rec)
        return log_p_inf + log_p_rec


class SIREmission(Emission[SIRState, InfectionObservation, Condition, SIRParameters]):
    """Poisson observation of new infections."""

    order: ClassVar[int] = 2
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[SIRState, SIRState],
        observation_history: tuple[()],
        condition: Condition,
        parameters: SIRParameters,
    ) -> InfectionObservation:
        prev_state, next_state = particle
        new_inf = prev_state.s - next_state.s
        obs = jrandom.poisson(key, jnp.maximum(new_inf, 0))
        return InfectionObservation(new_cases=obs)

    @staticmethod
    def log_prob(
        particle: tuple[SIRState, SIRState],
        observation_history: tuple[()],
        observation: InfectionObservation,
        condition: Condition,
        parameters: SIRParameters,
    ) -> Scalar:
        prev_state, next_state = particle
        new_inf = prev_state.s - next_state.s
        lam = jnp.maximum(new_inf, 0)
        return jstats.poisson.logpmf(observation.new_cases, lam)


class SIRModel(
    SequentialModel[SIRState, InfectionObservation, Condition, SIRParameters]
):
    """Sequential model wrapping the SIR components."""

    prior = SIRPrior()
    transition = SIRTransition()
    emission = SIREmission()
