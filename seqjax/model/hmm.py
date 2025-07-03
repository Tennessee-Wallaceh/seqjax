"""Discrete Hidden Markov Model components."""

from dataclasses import field
from typing import ClassVar

import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray, Scalar

from .base import Emission, Prior, SequentialModel, Transition
from .typing import Condition, Observation, Parameters, Particle


class HiddenState(Particle):
    """Latent state index for a discrete HMM."""

    z: Array


class DiscreteObservation(Observation):
    """Discrete observation index."""

    y: Array


class HMMParameters(Parameters):
    """Parameters of a discrete HMM."""

    initial_probs: Array = field(default_factory=lambda: jnp.ones(1))
    transition_matrix: Array = field(default_factory=lambda: jnp.ones((1, 1)))
    emission_probs: Array = field(default_factory=lambda: jnp.ones((1, 1)))


class CategoricalPrior(Prior[HiddenState, Condition, HMMParameters]):
    """Prior over the initial hidden state."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[Condition],
        parameters: HMMParameters,
    ) -> tuple[HiddenState]:
        state = jrandom.categorical(key, jnp.log(parameters.initial_probs))
        return (HiddenState(z=state),)

    @staticmethod
    def log_prob(
        particle: tuple[HiddenState],
        conditions: tuple[Condition],
        parameters: HMMParameters,
    ) -> Scalar:
        return jnp.log(parameters.initial_probs[particle[0].z])


class CategoricalTransition(Transition[HiddenState, Condition, HMMParameters]):
    """Categorical state transition."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[HiddenState],
        condition: Condition,
        parameters: HMMParameters,
    ) -> HiddenState:
        prev_state = particle_history[0].z
        logits = jnp.log(parameters.transition_matrix[prev_state])
        next_state = jrandom.categorical(key, logits)
        return HiddenState(z=next_state)

    @staticmethod
    def log_prob(
        particle_history: tuple[HiddenState],
        particle: HiddenState,
        condition: Condition,
        parameters: HMMParameters,
    ) -> Scalar:
        prev_state = particle_history[0].z
        return jnp.log(parameters.transition_matrix[prev_state, particle.z])


class CategoricalEmission(
    Emission[HiddenState, DiscreteObservation, Condition, HMMParameters]
):
    """Emission distribution conditioned on the hidden state."""

    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[HiddenState],
        observation_history: tuple[()],
        condition: Condition,
        parameters: HMMParameters,
    ) -> DiscreteObservation:
        state = particle[0].z
        logits = jnp.log(parameters.emission_probs[state])
        obs = jrandom.categorical(key, logits)
        return DiscreteObservation(y=obs)

    @staticmethod
    def log_prob(
        particle: tuple[HiddenState],
        observation_history: tuple[()],
        observation: DiscreteObservation,
        condition: Condition,
        parameters: HMMParameters,
    ) -> Scalar:
        state = particle[0].z
        return jnp.log(parameters.emission_probs[state, observation.y])


class HiddenMarkovModel(
    SequentialModel[HiddenState, DiscreteObservation, Condition, HMMParameters]
):
    """Discrete Hidden Markov Model."""

    particle_type = HiddenState
    prior = CategoricalPrior()
    transition = CategoricalTransition()
    emission = CategoricalEmission()
