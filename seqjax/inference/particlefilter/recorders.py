from __future__ import annotations

from typing import Callable, Sequence, cast

from jaxtyping import Array, PyTree

from .base import Recorder

import jax.numpy as jnp

from seqjax.model.base import ParticleType


def current_particle_mean(
    extractor: Callable[[ParticleType], Array],
) -> Recorder:
    """Return a recorder capturing the mean of ``extractor`` over particles."""

    def _recorder(
        weights: Array,
        particles: tuple[ParticleType, ...],
        _ancestors: Array,
        _obs: object,
        _cond: object,
        _last_particles: tuple[ParticleType, ...],
        _last_log_w: Array,
        _log_weight_sum: Array,
        _ess: Array,
    ) -> PyTree:
        current = extractor(particles[-1])
        expanded = jnp.reshape(weights, weights.shape + (1,) * (current.ndim - 1))
        return jnp.sum(current * expanded, axis=0)

    return cast(Recorder, _recorder)


def _weighted_quantiles(values: Array, weights: Array, qs: Array) -> Array:
    """Helper computing weighted quantiles along axis ``0``."""
    order = jnp.argsort(values, axis=0)
    sorted_vals = jnp.take_along_axis(values, order, axis=0)
    sorted_weights = jnp.take_along_axis(weights[:, None], order, axis=0)

    cum_weights = jnp.cumsum(sorted_weights, axis=0)
    cum_weights = cum_weights / cum_weights[-1]

    def _interp(q: Array) -> Array:
        idx = jnp.argmax(cum_weights >= q, axis=0)
        return jnp.take_along_axis(sorted_vals, idx[None, :], axis=0)[0]

    return jnp.stack([_interp(q) for q in qs])


def current_particle_quantiles(
    extractor: Callable[[ParticleType], Array],
    quantiles: Sequence[float] = (0.1, 0.9),
) -> Recorder:
    """Return a recorder capturing ``quantiles`` of ``extractor`` over particles."""

    qs = jnp.array(quantiles)

    def _recorder(
        weights: Array,
        particles: tuple[ParticleType, ...],
        _ancestors: Array,
        _obs: object,
        _cond: object,
        _last_particles: tuple[ParticleType, ...],
        _last_log_w: Array,
        _log_weight_sum: Array,
        _ess: Array,
    ) -> PyTree:
        current = extractor(particles[-1])
        flat = current.reshape(current.shape[0], -1)
        q_vals = _weighted_quantiles(flat, weights, qs)
        return q_vals.reshape((qs.shape[0],) + current.shape[1:])

    return cast(Recorder, _recorder)


def current_particle_variance(
    extractor: Callable[[ParticleType], Array],
) -> Recorder:
    """Return a recorder capturing the weighted variance of ``extractor``."""

    def _recorder(
        weights: Array,
        particles: tuple[ParticleType, ...],
        _ancestors: Array,
        _obs: object,
        _cond: object,
        _last_particles: tuple[ParticleType, ...],
        _last_log_w: Array,
        _log_weight_sum: Array,
        _ess: Array,
    ) -> PyTree:
        current = extractor(particles[-1])
        expanded = jnp.reshape(weights, weights.shape + (1,) * (current.ndim - 1))
        mean = jnp.sum(current * expanded, axis=0)
        return jnp.sum(expanded * (current - mean) ** 2, axis=0)

    return cast(Recorder, _recorder)


def log_marginal() -> Recorder:
    """Record the log marginal likelihood estimate at each step."""

    def _recorder(
        _weights: Array,
        _particles: tuple[ParticleType, ...],
        _ancestors: Array,
        _obs: object,
        _cond: object,
        _last_particles: tuple[ParticleType, ...],
        _last_log_w: Array,
        log_weight_sum: Array,
        _ess: Array,
    ) -> PyTree:
        return log_weight_sum - jnp.log(_weights.shape[0])

    return cast(Recorder, _recorder)


def effective_sample_size() -> Recorder:
    """Record the effective sample size at each step."""

    def _recorder(
        _weights: Array,
        _particles: tuple[ParticleType, ...],
        _ancestors: Array,
        _obs: object,
        _cond: object,
        _last_particles: tuple[ParticleType, ...],
        _last_log_w: Array,
        _log_weight_sum: Array,
        ess_val: Array,
    ) -> PyTree:
        return ess_val

    return cast(Recorder, _recorder)
