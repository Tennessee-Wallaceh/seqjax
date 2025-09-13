from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .base import FilterData


def _normalised_weights(log_w: Array) -> Array:
    """Convert log weights to normalised weights."""
    lw_max = jnp.max(log_w)
    w = jnp.exp(log_w - lw_max)
    return w / jnp.sum(w)


def current_particle_mean(filter_data: FilterData) -> PyTree:
    """Record the weighted mean of the current particles."""

    weights = _normalised_weights(filter_data.log_w)
    particle: PyTree = filter_data.particles[-1]

    def _mean(arr: Array) -> Array:
        expanded = jnp.reshape(weights, weights.shape + (1,) * (arr.ndim - 1))
        return jnp.sum(arr * expanded, axis=0)

    return jax.tree_util.tree_map(_mean, particle)


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
    filter_data: FilterData,
    *,
    quantiles: Sequence[float] = (0.1, 0.9),
) -> PyTree:
    """Record ``quantiles`` of the current particles."""

    weights = _normalised_weights(filter_data.log_w)
    particle: PyTree = filter_data.particles[-1]
    qs = jnp.array(quantiles)

    def _quant(arr: Array) -> Array:
        flat = arr.reshape(arr.shape[0], -1)
        q_vals = _weighted_quantiles(flat, weights, qs)
        return q_vals.reshape((qs.shape[0],) + arr.shape[1:])

    return jax.tree_util.tree_map(_quant, particle)


def current_particle_variance(filter_data: FilterData) -> PyTree:
    """Record the weighted variance of the current particles."""

    weights = _normalised_weights(filter_data.log_w)
    particle: PyTree = filter_data.particles[-1]

    def _var(arr: Array) -> Array:
        expanded = jnp.reshape(weights, weights.shape + (1,) * (arr.ndim - 1))
        mean = jnp.sum(arr * expanded, axis=0)
        return jnp.sum(expanded * (arr - mean) ** 2, axis=0)

    return jax.tree_util.tree_map(_var, particle)


def log_marginal(filter_data: FilterData) -> PyTree:
    """Record the log marginal likelihood increment at each step."""

    log_w_increment = filter_data.log_weight_increment
    lw_max = jnp.max(log_w_increment)
    w = jnp.exp(log_w_increment - lw_max)
    w_sum = jnp.sum(w)
    return jnp.log(w_sum) + lw_max - jnp.log(filter_data.ancestor_ix.shape[0])


def effective_sample_size(filter_data: FilterData) -> PyTree:
    """Record the effective sample size at each step."""

    return filter_data.ess_e
