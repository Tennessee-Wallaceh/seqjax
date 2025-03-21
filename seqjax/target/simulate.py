from typing import NamedTuple, TypeVar, Protocol, Generic, Optional
from jaxtyping import Array, Float, PRNGKeyArray
import jax.random as jrandom
from seqjax.target.base import Target, Particle, Observation, Condition, Hyperparameters

import jax
import jax.numpy as jnp


def simulate(
    key: PRNGKeyArray,
    target: Target[Particle, Observation, Condition, Hyperparameters],
    condition: Condition,
    hyperparameters: Hyperparameters,
    num_steps: int,
):
    init_key, *step_keys = jrandom.split(key, num_steps + 1)
    x_0 = target.sample_prior(init_key, hyperparameters)
    reference_emission = target.reference_emission(hyperparameters)

    def step(state, inputs):
        step_key, condition = inputs
        particle, last_emission = state

        condition = target.emission_to_condition(
            last_emission, condition, hyperparameters
        )

        transition_key, emission_key = jrandom.split(step_key)

        next_particle = target.sample_transition(
            transition_key, particle, condition, hyperparameters
        )
        emission = target.sample_emission(
            emission_key, next_particle, condition, hyperparameters
        )
        return (next_particle, emission), (next_particle, emission, condition)

    _, (latent_path, observed_path, condition_path) = jax.lax.scan(
        step,
        (x_0, reference_emission),
        xs=(jnp.array(step_keys), condition),
        length=num_steps,
    )
    return latent_path, observed_path, condition_path
