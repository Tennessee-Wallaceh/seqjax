from __future__ import annotations

from typing import Callable, Sequence

import jax.numpy as jnp
from jaxtyping import Array

from seqjax.model.base import ParticleType


def current_particle_mean(
    extractor: Callable[[ParticleType], Array],
) -> Callable[[Array, tuple[ParticleType, ...]], Array]:
    """Return a recorder capturing the mean of ``extractor`` over particles."""

    def _recorder(weights: Array, particles: tuple[ParticleType, ...]) -> Array:
        current = extractor(particles[-1])
        expanded = jnp.reshape(weights, weights.shape + (1,) * (current.ndim - 1))
        return jnp.sum(current * expanded, axis=0)

    return _recorder


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
) -> Callable[[Array, tuple[ParticleType, ...]], Array]:
    """Return a recorder capturing ``quantiles`` of ``extractor`` over particles."""

    qs = jnp.array(quantiles)

    def _recorder(weights: Array, particles: tuple[ParticleType, ...]) -> Array:
        current = extractor(particles[-1])
        flat = current.reshape(current.shape[0], -1)
        q_vals = _weighted_quantiles(flat, weights, qs)
        return q_vals.reshape((qs.shape[0],) + current.shape[1:])

    return _recorder


def current_particle_variance(
    extractor: Callable[[ParticleType], Array],
) -> Callable[[Array, tuple[ParticleType, ...]], Array]:
    """Return a recorder capturing the weighted variance of ``extractor``."""

    def _recorder(weights: Array, particles: tuple[ParticleType, ...]) -> Array:
        current = extractor(particles[-1])
        expanded = jnp.reshape(weights, weights.shape + (1,) * (current.ndim - 1))
        mean = jnp.sum(current * expanded, axis=0)
        return jnp.sum(expanded * (current - mean) ** 2, axis=0)

    return _recorder
