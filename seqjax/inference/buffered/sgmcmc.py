import typing
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
from .buffered import _run_segment
from ..sgld import SGLDConfig, run_sgld


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
) -> tuple[typing.Callable[[ParametersType, PRNGKeyArray], ParametersType], int]:
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

    def log_post(
        params: ParametersType, start: jax.Array, pf_key: PRNGKeyArray
    ) -> jax.Array:
        log_mps = _run_segment(
            start,
            smc,
            pf_key,
            params,
            observations,
            condition_path,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
        )
        return prior.log_prob(params, config.hyperparameters) + jnp.sum(log_mps)

    def grad_estimator(params: ParametersType, key: PRNGKeyArray) -> ParametersType:
        start_key, pf_key = jrandom.split(key)
        start = jrandom.randint(start_key, (), 0, start_max)
        return jax.grad(lambda p: log_post(p, start, pf_key))(params)

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
