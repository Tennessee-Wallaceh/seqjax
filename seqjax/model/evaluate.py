"""Model evaluation utilities for computing log probabilities."""

import jax
import jax.numpy as jnp
from jaxtyping import Scalar

from seqjax.model.base import (
    SequentialModel,
)
from seqjax.util import (
    concat_pytree,
    index_pytree,
    pytree_shape,
    slice_pytree,
)


def log_prob_x(
    target: SequentialModel,
    x_path,
    condition,
    parameters,
    *,
    x_history=None,
) -> Scalar:
    """Return ``log p(x)`` for a latent sequence.

    ``x_path`` should contain only the ``t \\geq 0`` portion of the latent
    sequence.  If ``target.prior.order > 1`` then the required history prior to
    ``t=0`` can be supplied via ``x_history``.

    Internally the function slices out the latent histories for each time step to
    allow a vectorised evaluation of the density.  This trades memory for speed
    and may require a more memory-efficient implementation for long sequences.
    """
    if x_history is not None:
        x_path = concat_pytree(x_history, x_path)

    sequence_start = target.prior.order - 1
    x_shape = x_path.batch_shape
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
    transition_condition = slice_pytree(
        condition,
        sequence_start + 1,
        sequence_start + sequence_length,
    )

    transition_log_p_x = jax.vmap(target.transition.log_prob, in_axes=[0, 0, 0, None])(
        particle_history,
        target_particle,
        transition_condition,
        parameters,
    ).sum()

    return (log_p_x_0 + transition_log_p_x).sum()


def log_prob_y_given_x(
    target: SequentialModel,
    x_path,
    y_path,
    condition,
    parameters,
    *,
    x_history=None,
) -> Scalar:
    """Return ``log p(y | x)`` for a sequence of observations.

    ``x_path`` should again only contain the latent states for ``t \\geq 0``.  If
    the model requires additional latent history this should be provided via
    ``x_history``.  The observation path ``y_path`` is assumed to contain any
    required observation history already.

    ``x_path`` and ``y_path`` share the same leading ``Batch`` dimensions,
    matching the output of :func:`~seqjax.model.simulate.simulate`.
    """
    if x_history is not None:
        x_path = concat_pytree(x_history, x_path)

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
    *,
    x_history=None,
) -> Scalar:
    """Return ``log p(x, y)`` for a path and observations.

    ``x_path`` should contain only the ``t \\geq 0`` portion of the latent path.
    Pass ``x_history`` to supply any earlier latent values required by
    ``target.prior`` or the emission model.

    The latent and observation sequences share their ``Batch`` dimensions,
    reflecting the output of :func:`~seqjax.model.simulate.simulate`.
    """
    return log_prob_x(
        target,
        x_path,
        condition,
        parameters,
        x_history=x_history,
    ) + log_prob_y_given_x(
        target,
        x_path,
        y_path,
        condition,
        parameters,
        x_history=x_history,
    )


def get_log_prob_x_for_target(
    target: SequentialModel,
):
    """Return a ``log_prob_x`` function bound to ``target``."""

    def _log_prob_x(
        x_path,
        condition,
        parameters,
        *,
        x_history=None,
    ):
        return log_prob_x(target, x_path, condition, parameters, x_history=x_history)

    return _log_prob_x


def get_log_prob_joint_for_target(
    target: SequentialModel,
):
    """Return a ``log_prob_joint`` function bound to ``target``."""

    def _log_prob_joint(
        x_path,
        y_path,
        condition,
        parameters,
        *,
        x_history=None,
    ):
        return log_prob_joint(
            target,
            x_path,
            y_path,
            condition,
            parameters,
            x_history=x_history,
        )

    return _log_prob_joint


def buffered_log_p_x(
    target,
    x_path,
    condition,
    parameters,
) -> Scalar:
    """
    slice out particle histories for vectorised evaluation
    trade off here is copy+vectorised vs no copy+sequential evaluation
    for longer sequences expect copy+vector to be faster, but requires more memory
    for very large sequences + order + batch sizes, this could trigger OOM, and
    smarter implementation could be required
    """
    sequence_start = target.prior.order - 1
    sequence_length = x_path.batch_shape[0] - sequence_start

    # compute prior
    prior_particles = tuple(index_pytree(x_path, i) for i in range(target.prior.order))
    prior_params = index_pytree(parameters, 0)
    prior_conditions = tuple(
        index_pytree(condition, i) for i in range(target.prior.order)
    )

    log_p_x_0 = target.prior.log_prob(prior_particles, prior_conditions, prior_params)

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
        x_path, sequence_start + 1, sequence_start + sequence_length
    )

    transition_condition = slice_pytree(condition, sequence_start + 1, sequence_length)
    transition_parameters = slice_pytree(
        parameters, sequence_start + 1, sequence_length
    )

    transition_log_p_x = jax.vmap(target.transition.log_prob)(
        particle_history, target_particle, transition_condition, transition_parameters
    )

    return jnp.hstack([log_p_x_0, transition_log_p_x])


def buffered_log_p_y_given_x(
    target,
    x_path,
    y_path,
    condition,
    parameters,
) -> Scalar:
    x_length = x_path.batch_shape[0]

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

    return jax.vmap(target.emission.log_prob, in_axes=[0, 0, 0, 0, 0])(
        particle_history,
        emission_history,
        observations,
        observation_conditions,
        parameters,
    )


def buffered_log_p_joint(
    target,
    x_path,
    y_path,
    condition,
    parameters,
) -> Scalar:
    out = buffered_log_p_y_given_x(target, x_path, y_path, condition, parameters)
    out += buffered_log_p_x(target, x_path, condition, parameters)
    return out
