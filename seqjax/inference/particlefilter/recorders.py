from __future__ import annotations

from typing import Callable, Sequence, cast

from jaxtyping import Array, PyTree

from .base import Recorder, FilterData

import jax.numpy as jnp

from seqjax.model.base import ParticleType


def _normalised_weights(log_w: Array) -> Array:
    """Convert log weights to normalised weights."""
    lw_max = jnp.max(log_w)
    w = jnp.exp(log_w - lw_max)
    return w / jnp.sum(w)


def current_particle_mean(
    extractor: Callable[[ParticleType], Array],
) -> Recorder:
    """Return a recorder capturing the mean of ``extractor`` over particles."""

    def _recorder(filter_data: FilterData) -> PyTree:
        weights = _normalised_weights(filter_data.log_w)
        current = extractor(filter_data.particles[-1])
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

    def _recorder(filter_data: FilterData) -> PyTree:
        weights = _normalised_weights(filter_data.log_w)
        current = extractor(filter_data.particles[-1])
        flat = current.reshape(current.shape[0], -1)
        q_vals = _weighted_quantiles(flat, weights, qs)
        return q_vals.reshape((qs.shape[0],) + current.shape[1:])

    return cast(Recorder, _recorder)


def current_particle_variance(
    extractor: Callable[[ParticleType], Array],
) -> Recorder:
    """Return a recorder capturing the weighted variance of ``extractor``."""

    def _recorder(filter_data: FilterData) -> PyTree:
        weights = _normalised_weights(filter_data.log_w)
        current = extractor(filter_data.particles[-1])
        expanded = jnp.reshape(weights, weights.shape + (1,) * (current.ndim - 1))
        mean = jnp.sum(current * expanded, axis=0)
        return jnp.sum(expanded * (current - mean) ** 2, axis=0)

    return cast(Recorder, _recorder)


def log_marginal() -> Recorder:
    """Record the log marginal likelihood estimate at each step."""

    def _recorder(filter_data) -> PyTree:
        log_w_increment = filter_data.log_weight_increment
        lw_max = jnp.max(log_w_increment)
        w = jnp.exp(log_w_increment - lw_max)
        w_sum = jnp.sum(w)
        log_marg_inc = (
            jnp.log(w_sum) + lw_max - jnp.log(filter_data.ancestor_ix.shape[0])
        )
        return log_marg_inc

    return cast(Recorder, _recorder)


def effective_sample_size() -> Recorder:
    """Record the effective sample size at each step."""

    def _recorder(filter_data: FilterData) -> PyTree:
        return filter_data.ess_e

    return cast(Recorder, _recorder)
