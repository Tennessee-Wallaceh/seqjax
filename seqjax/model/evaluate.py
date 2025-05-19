import jax
from jaxtyping import Scalar, PyTree
from seqjax.util import slice_pytree, index_pytree, pytree_shape, concat_pytree

from seqjax.model.base import (
    Target,
    ParticleType,
    ObservationType,
    ConditionType,
    ParametersType,
)


# TODO: is this use of PyTree correct?
def log_p_x(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
    x_path: PyTree[ParticleType, "sequence_length"],
    condition: PyTree[ConditionType, "sequence_length"],
    parameters: ParametersType,
) -> Scalar:
    """
    slice out particle histories for vectorised evaluation
    trade off here is copy+vectorised vs no copy+sequential evaluation
    for longer sequences expect copy+vector to be faster, but requires more memory
    for very large sequences + order + batch sizes, this could trigger OOM, and 
    smarter implementation could be required
    """
    sequence_start = target.prior.order - 1
    sequence_length = pytree_shape(x_path)[0] - sequence_start

    # compute prior
    prior_particles = tuple(
        index_pytree(x_path, i)
        for i in range(target.prior.order)
    )
    prior_conditions = tuple(
        index_pytree(condition, i) 
        for i in range(target.prior.order)
    )

    log_p_x_0 = target.prior.log_p(prior_particles, prior_conditions, parameters)

    # rest of sequence
    particle_history = tuple(
        slice_pytree(
            x_path, 
            sequence_start + 1 + i, # starting from t = seq_start + 1
            sequence_start + i + sequence_length
        )
        for i in range(-target.transition.order, 0)
    )
    target_particle = slice_pytree(
        x_path, 
        sequence_start + 1, 
        sequence_start + sequence_length
    )
    transition_condition = slice_pytree(condition, sequence_start, sequence_length)

    transition_log_p_x = jax.vmap(target.transition.log_p, in_axes=[0, 0, 0, None])(
        particle_history, 
        target_particle, 
        transition_condition, 
        parameters
    ).sum()

    return (log_p_x_0 + transition_log_p_x).sum()


def log_p_x_noncentered(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
    eps_path: PyTree[ParticleType, "sequence_length"],
    condition: PyTree[ConditionType, "sequence_length"],
    parameters: ParametersType,
):
    sequence_start = target.prior.order - 1
    sequence_length = pytree_shape(eps_path)[0] - sequence_start

    prior_particles = tuple(
        index_pytree(eps_path, i)
        for i in range(target.prior.order)
    )
    prior_conditions = tuple(index_pytree(condition, ix) for ix in range(target.prior.order))

    log_p_x_0 = target.prior.log_p(prior_particles, prior_conditions, parameters)

    def body(particle_history, inputs):
        eps_t, cond_t = inputs
        loc, scale = target.transition.loc_scale(particle_history, cond_t, parameters)
        lp_innov  = target.transition.log_p_innovation(eps_t, loc, scale)
        next_particle = target.transition.apply_innovation(eps_t, loc, scale)
        return (*particle_history[1:], next_particle), (next_particle, lp_innov)

    particle_history = prior_particles[-target.transition.order:]
    # inputs = index_pytree(eps_path, 0), index_pytree(condition, 0)

    _, (e_x_path, log_p_eps) = jax.lax.scan(
        body, 
        particle_history, 
        (
            slice_pytree(eps_path, sequence_start + 1, sequence_start + sequence_length), 
            slice_pytree(condition, sequence_start + 1, sequence_start + sequence_length), 
        )
    )
    
    x_path = concat_pytree(*prior_particles, e_x_path)
    return log_p_x_0 + log_p_eps.sum(), x_path

def log_p_y_given_x(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
    x_path: PyTree[ParticleType, "x_length"],
    y_path: PyTree[ParticleType, "y_length"],
    condition: PyTree[ConditionType, "y_length"],
    parameters: ParametersType,
) -> Scalar:
    x_length = pytree_shape(x_path)[0]

    x_sequence_start = target.prior.order - 1
    y_sequence_start = target.emission.observation_dependency

    # this is the length of the observation sequence
    # should == y_sequence_length - y_sequence_start
    sequence_length = x_length - x_sequence_start
    
    particle_history = tuple(
        slice_pytree(x_path, x_sequence_start + i, x_sequence_start + i + sequence_length)
        for i in range(-target.emission.order + 1, 1)
    )

    emission_history = tuple(
        slice_pytree(y_path, y_sequence_start + i, y_sequence_start + i + sequence_length)
        for i in range(-target.emission.observation_dependency, 0)
    )

    observations = slice_pytree(y_path, y_sequence_start, sequence_length + y_sequence_start)
    observation_conditions = slice_pytree(condition, y_sequence_start, sequence_length + y_sequence_start)

    return jax.vmap(
        target.emission.log_p, 
        in_axes=[0, 0, 0, 0, None]
    )(
        particle_history,
        emission_history,
        observations,
        observation_conditions, 
        parameters
    ).sum()


def log_p_joint(
    target,
    x_path,
    y_path,
    condition,
    parameters,
) -> Scalar:
    return log_p_x(
        target,
        x_path,
        condition,
        parameters
    ) + log_p_y_given_x(
        target,
        x_path,
        y_path,
        condition,
        parameters
    )


# some utilities for getting densities for specific target
def get_log_p_x_for_target(
    target: Target[ParticleType, ObservationType, ConditionType,  ParametersType],
):
    def _log_p_x(
        x_path: PyTree[ParticleType, "num_steps"],
        condition: PyTree[ConditionType, "num_steps"],
        parameters: ParametersType,
    ):
        return log_p_x(target, x_path, condition, parameters)

    return _log_p_x


def get_log_p_joint_for_target(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
):
    def _log_p_joint(
        x_path,
        y_path,
        condition,
        parameters,
    ):
        return log_p_joint(target, x_path, y_path, condition, parameters)

    return _log_p_joint
