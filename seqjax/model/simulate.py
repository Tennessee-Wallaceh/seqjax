import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray, PyTree
from functools import partial
from typing import Optional
from seqjax.util import index_pytree, slice_pytree, concat_pytree
from seqjax.model.base import (
    Target,
    ParticleType,
    ObservationType,
    ConditionType,
    ParametersType,
)

def step(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType], 
    parameters: ParametersType, 
    state: tuple[
        tuple[ParticleType, ...],
        tuple[ObservationType, ...],
    ], 
    inputs
):
    # pad the RHS with a None, then no condition => condition is None
    step_key, condition = (inputs + (None,))[:2]
    particles, emissions = state
    transition_key, emission_key = jrandom.split(step_key)

    # last particle is at t
    # sample x_t+1 then y_t+1
    next_particle = target.transition.sample(
        transition_key, 
        particles[-target.transition.order:],
        condition, 
        parameters
    )

    particles = (*particles, next_particle)

    emission = target.emission.sample(
        emission_key, 
        particles[-target.emission.order:],
        emissions, 
        condition, 
        parameters
    )

    # add the next particle to the history before emission
    # only pass on necessary information
    # # read off histories of appropriate order
    max_latent_order = max(target.transition.order, target.emission.order)
    particle_history = (*particles, next_particle)[-max_latent_order:]
    emission_history = (*emissions, emission)
    emission_history = emission_history[len(emission_history)-target.emission.observation_dependency:]

    return (particle_history, emission_history), (next_particle, emission)

def simulate(
    key: PRNGKeyArray,
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
    condition: Optional[PyTree[ConditionType, "sequence_length"]],
    parameters: ParametersType,
    sequence_length: int,
):
    init_x_key, init_y_key, *step_keys = jrandom.split(key, sequence_length + 1)
    
    # special handling for sampling first state
    prior_conditions = tuple(index_pytree(condition, ix) for ix in range(target.prior.order))
    condition_0 = index_pytree(condition, target.prior.order - 1)
    x_0 = target.prior.sample(init_x_key, prior_conditions, parameters)
    y_0 = target.emission.sample(
        init_y_key, 
        x_0,
        parameters.reference_emission, 
        condition_0, 
        parameters
    )

    # build start point of scan
    emission_history = (*parameters.reference_emission, y_0)
    emission_history = tuple(emission_history[len(emission_history)-target.emission.observation_dependency:])
    state = (x_0, emission_history)
    inputs = (
        (jnp.array(step_keys),) 
        if condition is None 
        else (
            jnp.array(step_keys), 
            slice_pytree(
                condition, 
                target.prior.order, 
                sequence_length + target.prior.order - 1
            )
        )
    )

    # scan for generic step
    _, (latent_path, observed_path) = jax.lax.scan(
        partial(step, target, parameters),
        state,
        xs=inputs,
        length=sequence_length - 1,
    )

    latent_path = concat_pytree(*x_0, latent_path)
    observed_path = concat_pytree(*parameters.reference_emission, y_0, observed_path)
    return latent_path, observed_path