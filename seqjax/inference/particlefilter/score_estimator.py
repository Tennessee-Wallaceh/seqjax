"""
Implement estimators in
Particle Approximations of the Score and Observed
Information Matrix for Parameter Estimation in State Space Models With Linear
Computational Cost

https://eprints.whiterose.ac.uk/id/eprint/83198/9/2_pdfsam_J_of_Comp_Graph_Stats_2015.pdf
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
)
from seqjax.model.typing import Batched, SequenceAxis
from .base import SMCSampler, run_filter
from seqjax.util import dynamic_index_pytree_in_dim


def _step_joint(
    target,
    parameters,
    particle_history,
    particle,
    observation_history,
    observation,
    condition,
):
    return target.emission.log_prob(
        (particle,), observation_history, observation, condition, parameters
    ) + target.transition.log_prob(particle_history, particle, condition, parameters)

def _gather_pytree_batch(tree, indices):
    return jax.vmap(lambda i: dynamic_index_pytree_in_dim(tree, i, 0))(indices)


def _weighted_mean(tree, weights):
    def _mean(x):
        w = jnp.reshape(weights, weights.shape + (1,) * (x.ndim - 1))
        return jnp.sum(x * w, axis=0)

    return jax.tree_util.tree_map(_mean, tree)


def _stack_tree(xs):
    return jax.tree_util.tree_map(lambda *l: jnp.stack(l), *xs)


def run_score_estimator(
    particle_filter: SMCSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    parameters: ParametersType,
    observation_path: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    *,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_history: tuple[ObservationType, ...] | None = None,
) -> tuple[ParametersType, Batched[ParametersType, SequenceAxis | int]]:
    """Estimate the score of ``observation_path`` under ``parameters``."""

    target = particle_filter.target

    step_score = jax.grad(
        lambda p, ph, pa, oh, ob, c: _step_joint(
            target, p, ph, pa, oh, ob, c
        )
    )

    step_score_vmap = jax.vmap(
        step_score, in_axes=(None, 0, 0, None, None, None)
    )

    def _recorder(
        weights,
        particles,
        ancestors,
        observation,
        condition,
        last_particles,
        _last_log_w,
        _log_weight_sum,
        _ess,
    ):
        parents = _gather_pytree_batch(last_particles[-1], ancestors)
        grads = step_score_vmap(
            parameters,
            (parents,),
            particles[-1],
            (),
            observation,
            condition,
        )
        return grads, weights

    log_w, _p, ancestor_hist, (rec_hist,) = run_filter(
        particle_filter,
        key,
        parameters,
        observation_path,
        condition_path,
        initial_conditions=initial_conditions,
        observation_history=observation_history,
        recorders=(_recorder,),
    )
    grad_hist, weight_hist = rec_hist

    def _zeros_like(x):
        return jnp.zeros_like(x[0])

    score = jax.tree_util.tree_map(_zeros_like, grad_hist)
    particles_score = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x[0]), grad_hist
    )
    steps = []

    for t in range(ancestor_hist.shape[0]):
        g_t = jax.tree_util.tree_map(lambda x: x[t], grad_hist)
        w_t = weight_hist[t]
        a_t = ancestor_hist[t]
        particles_score = _gather_pytree_batch(particles_score, a_t)
        particles_score = jax.tree_util.tree_map(
            lambda a, b: a + b, particles_score, g_t
        )
        step = _weighted_mean(particles_score, w_t)
        particles_score = jax.tree_util.tree_map(
            lambda a, s: a - s, particles_score, step
        )
        score = jax.tree_util.tree_map(lambda s, d: s + d, score, step)
        steps.append(step)

    step_hist = _stack_tree(steps)
    return score, step_hist
