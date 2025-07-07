from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
    ParameterPrior,
    HyperParametersType,
)
from seqjax.model.typing import Batched, SequenceAxis, SampleAxis
from seqjax.inference.particlefilter import SMCSampler
from .buffered import _pad_edges_pytree
from seqjax.util import dynamic_slice_pytree, dynamic_index_pytree_in_dim
from ..sgld import SGLDConfig, run_sgld
from ..particlefilter.score_estimator import run_score_estimator


class BufferedSGLDConfig(eqx.Module):
    """Configuration for :func:`run_buffered_sgld`."""

    buffer_size: int = 0
    batch_size: int = 1
    particle_filter: (
        SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType] | None
    ) = None
    parameter_prior: ParameterPrior[ParametersType, HyperParametersType] | None = None
    hyperparameters: HyperParametersType | None = None


def _make_grad_estimator(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    observations: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None,
    config: BufferedSGLDConfig,
) -> tuple[Callable[[ParametersType, PRNGKeyArray], ParametersType], int]:
    """Return gradient estimator and maximum start index."""

    smc = config.particle_filter
    if smc is None:
        raise ValueError("particle_filter must be provided in config")
    if smc.target is not target:
        smc = eqx.tree_at(lambda m: m.target, smc, target)

    prior = config.parameter_prior
    if prior is None:
        raise ValueError("parameter_prior must be provided in config")

    seq_len = jax.tree_util.tree_leaves(observations)[0].shape[0]
    if config.batch_size > seq_len:
        raise ValueError("batch_size must not exceed sequence length")

    start_max = seq_len - config.batch_size + 1

    def _score_segment(
        start: jax.Array,
        pf_key: PRNGKeyArray,
        params: ParametersType,
    ) -> ParametersType:
        slice_size = config.batch_size + 2 * config.buffer_size

        obs_padded = _pad_edges_pytree(observations, config.buffer_size)
        if condition_path is not None:
            cond_padded = _pad_edges_pytree(condition_path, config.buffer_size)
        else:
            cond_padded = None

        obs_slice = dynamic_slice_pytree(obs_padded, start, slice_size)
        if condition_path is not None:
            cond_slice = dynamic_slice_pytree(cond_padded, start, slice_size)
        else:
            cond_slice = None

        if smc.target.prior.order > 0:
            init_conds = tuple(
                dynamic_index_pytree_in_dim(cond_padded, start + i, 0)
                if cond_padded is not None
                else None
                for i in range(smc.target.prior.order)
            )
        else:
            init_conds = ()

        score, step_hist = run_score_estimator(
            smc,
            pf_key,
            params,
            obs_slice,
            cond_slice,
            initial_conditions=init_conds,
        )

        return jax.tree_util.tree_map(
            lambda x: jnp.sum(
                x[config.buffer_size : config.buffer_size + config.batch_size],
                axis=0,
            ),
            step_hist,
        )

    def grad_estimator(params: ParametersType, key: PRNGKeyArray) -> ParametersType:
        start_key, pf_key = jrandom.split(key)
        start = jrandom.randint(start_key, (), 0, start_max)
        like_grad = _score_segment(start, pf_key, params)
        prior_grad = jax.grad(prior.log_prob)(params, config.hyperparameters)
        return jax.tree_util.tree_map(lambda a, b: a + b, like_grad, prior_grad)

    return grad_estimator, start_max


def run_buffered_sgld(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    parameters: ParametersType,
    observations: Batched[ObservationType, SequenceAxis],
    *,
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    config: BufferedSGLDConfig,
    sgld_config: SGLDConfig = SGLDConfig(),
) -> Batched[ParametersType, SampleAxis | int]:
    """Run buffered SGLD updates over ``observations``."""

    grad_estimator, _ = _make_grad_estimator(
        target, observations, condition_path, config
    )
    return run_sgld(grad_estimator, key, parameters, config=sgld_config)
