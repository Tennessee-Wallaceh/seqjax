import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray, PyTree
from functools import partial

from seqjax.target.base import (
    Target,
    ParticleType,
    ObservationType,
    ConditionType,
    ParametersType,
)

def step_with_emission(target, parameters, state, inputs):
    step_key, condition = inputs
    particle, last_emission = state

    condition = target.emission_to_condition(last_emission, condition, parameters)

    transition_key, emission_key = jrandom.split(step_key)

    next_particle = target.transition.sample(
        transition_key, particle, condition, parameters
    )
    emission = target.emission.sample(
        emission_key, next_particle, condition, parameters
    )
    return (next_particle, emission), (next_particle, emission, condition)

def step_unconditional(target, parameters, particle, step_key):
    transition_key, emission_key = jrandom.split(step_key)

    next_particle = target.transition.sample(
        transition_key, particle, None, parameters
    )
    emission = target.emission.sample(
        emission_key, next_particle, None, parameters
    )
    return next_particle, (next_particle, emission)


def simulate(
    key: PRNGKeyArray,
    target: Target[ParticleType, ConditionType, ObservationType, ParametersType],
    condition: PyTree[ConditionType, "num_steps"],
    parameters: ParametersType,
    num_steps: int,
):
    init_key, *step_keys = jrandom.split(key, num_steps + 1)
    x_0 = target.prior.sample(init_key, parameters)
    reference_emission = target.reference_emission(parameters)

    if reference_emission is None:
        _, (latent_path, observed_path) = jax.lax.scan(
            partial(step_unconditional, target, parameters),
            x_0,
            xs=jnp.array(step_keys),
            length=num_steps,
        )
        condition_path = None
    else:
        _, (latent_path, observed_path, condition_path) = jax.lax.scan(
            partial(step_with_emission, target, parameters)
            (x_0, reference_emission),
            xs=(jnp.array(step_keys), condition),
            length=num_steps,
        )
    return latent_path, observed_path, condition_path
