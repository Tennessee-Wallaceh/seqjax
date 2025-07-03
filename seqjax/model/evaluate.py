"""Model evaluation utilities for computing log probabilities."""

import jax
from jaxtyping import PyTree, Scalar

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
)
from seqjax.util import index_pytree, pytree_shape, slice_pytree

# ``Target`` used to be an alias for ``SequentialModel``.  Define the
# alias here so that older annotations continue to import correctly.
Target = SequentialModel


def log_prob_x(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
    x_path: ParticleType,
    condition: ConditionType,
    parameters: ParametersType,
) -> Scalar:
    """Slice out particle histories for vectorised evaluation
    trade off here is copy+vectorised vs no copy+sequential evaluation
    for longer sequences expect copy+vector to be faster, but requires more memory
    for very large sequences + order + batch sizes, this could trigger OOM, and
    smarter implementation could be required
    """
    sequence_start = target.prior.order - 1
    x_shape = pytree_shape(x_path)[0]
    sequence_length = x_shape[0] - sequence_start

    # compute prior
    prior_particles = tuple(index_pytree(x_path, i) for i in range(target.prior.order))
    prior_conditions = tuple(
        index_pytree(condition, i) for i in range(target.prior.order)
    )

    log_p_x_0 = target.prior.log_prob(prior_particles, prior_conditions, parameters)

    # rest of sequence
    particle_history = tuple(
        slice_pytree(
            x_path,
            sequence_start + 1 + i,  # starting from t = seq_start + 1
            sequence_start + i + sequence_length,
        )
        for i in range(-target.transition.order, 0)
    )
    target_particle = slice_pytree(
        x_path,
        sequence_start + 1,
        sequence_start + sequence_length,
    )
    transition_condition = slice_pytree(condition, sequence_start, sequence_length)

    transition_log_p_x = jax.vmap(target.transition.log_prob, in_axes=[0, 0, 0, None])(
        particle_history,
        target_particle,
        transition_condition,
        parameters,
    ).sum()

    return (log_p_x_0 + transition_log_p_x).sum()

  
def log_prob_y_given_x(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
    x_path: PyTree,
    y_path: PyTree,
    condition: PyTree,
    parameters: ParametersType,
) -> Scalar:
    """Return ``log p(y | x)`` for a sequence of observations."""
    x_length = pytree_shape(x_path)[0][0]

    x_sequence_start = target.prior.order - 1
    y_sequence_start = target.emission.observation_dependency

    # this is the length of the observation sequence
    # should == y_sequence_length - y_sequence_start
    sequence_length = x_length - x_sequence_start

    particle_history = tuple(
        slice_pytree(
            x_path, x_sequence_start + i, x_sequence_start + i + sequence_length
        )
        for i in range(-target.emission.order + 1, 1)
    )

    emission_history = tuple(
        slice_pytree(
            y_path, y_sequence_start + i, y_sequence_start + i + sequence_length
        )
        for i in range(-target.emission.observation_dependency, 0)
    )

    observations = slice_pytree(
        y_path, y_sequence_start, sequence_length + y_sequence_start
    )
    observation_conditions = slice_pytree(
        condition, y_sequence_start, sequence_length + y_sequence_start
    )

    return jax.vmap(
        target.emission.log_prob,
        in_axes=[0, 0, 0, 0, None],
    )(
        particle_history,
        emission_history,
        observations,
        observation_conditions,
        parameters,
    ).sum()


def log_prob_joint(
    target,
    x_path,
    y_path,
    condition,
    parameters,
) -> Scalar:
    """Return ``log p(x, y)`` for a path and observations."""
    return log_prob_x(
        target,
        x_path,
        condition,
        parameters,
    ) + log_prob_y_given_x(
        target,
        x_path,
        y_path,
        condition,
        parameters,
    )

def get_log_prob_x_for_target(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
):
    """Return a ``log_prob_x`` function bound to ``target``."""

    def _log_prob_x(
        x_path: PyTree,
        condition: PyTree,
        parameters: ParametersType,
    ):
        return log_prob_x(target, x_path, condition, parameters)

    return _log_prob_x


def get_log_prob_joint_for_target(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
):
    """Return a ``log_prob_joint`` function bound to ``target``."""

    def _log_prob_joint(
        x_path,
        y_path,
        condition,
        parameters,
    ):
        return log_prob_joint(target, x_path, y_path, condition, parameters)

    return _log_prob_joint
