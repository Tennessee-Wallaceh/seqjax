import jax
from jaxtyping import Scalar, PyTree
from seqjax.util import slice_pytree, index_pytree, pytree_shape

from seqjax.target.base import (
    Target,
    ParticleType,
    ObservationType,
    ConditionType,
    ParametersType,
)


# TODO: is this use of PyTree correct?
def log_p_x(
    target: Target[ParticleType, ConditionType, ObservationType, ParametersType],
    x_path: PyTree[ParticleType, "num_steps"],
    condition: PyTree[ConditionType, "num_steps"],
    parameters: ParametersType,
) -> Scalar:
    num_steps = pytree_shape(x_path)
    particle = slice_pytree(x_path, 0, num_steps - 1)
    next_particle = slice_pytree(x_path, 1, num_steps)
    transition_condition = slice_pytree(condition, 1, num_steps)

    log_p_x0 = target.prior.log_p(index_pytree(x_path, 0), parameters)
    transition_log_p_x = jax.vmap(target.transition.log_p, in_axes=[0, 0, 0, None])(
        particle, next_particle, transition_condition, parameters
    )
    return log_p_x0 + transition_log_p_x.sum()


def log_p_y_given_x(
    target,
    x_path,
    y_path,
    condition,
    parameters,
) -> Scalar:
    return jax.vmap(target.emission.log_p, in_axes=[0, 0, 0, None])(
        x_path, y_path, condition, parameters
    ).sum()


def log_p_joint(
    target,
    x_path,
    y_path,
    condition,
    parameters,
) -> Scalar:
    return log_p_x(target, x_path, condition, parameters) + log_p_y_given_x(
        target,
        x_path,
        y_path,
        condition,
        parameters,
    )


# some utilities for getting densities for specific target
def get_log_p_x_for_target(
    target: Target[ParticleType, ConditionType, ObservationType, ParametersType],
):
    def _log_p_x(
        x_path: PyTree[ParticleType, "num_steps"],
        condition: PyTree[ConditionType, "num_steps"],
        parameters: ParametersType,
    ):
        return log_p_x(target, x_path, condition, parameters)

    return _log_p_x


def get_log_p_joint_for_target(
    target: Target[ParticleType, ConditionType, ObservationType, ParametersType],
):
    def _log_p_joint(
        x_path,
        y_path,
        condition,
        parameters,
    ):
        return log_p_joint(target, x_path, y_path, condition, parameters)

    return _log_p_joint
