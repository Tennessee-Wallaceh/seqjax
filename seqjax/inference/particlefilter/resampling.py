import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
from jaxtyping import Array, PRNGKeyArray

from seqjax.model import interface as model_interface
import seqjax.model.typing as seqjtyping
from seqjax.util import dynamic_index_pytree_in_dim as index_tree


class Resampler[
    ParticleT: seqjtyping.Latent,
    FilterLatentHistoryLength: int,
](typing.Protocol):
    """Map a weighted latent population to weighted ancestor histories."""

    def __call__(
        self,
        key: PRNGKeyArray,
        raw_log_weights: Array,
        particles: model_interface.LatentContext[
            ParticleT,
            FilterLatentHistoryLength,
        ],
        num_resample: int,
    ) -> tuple[
        model_interface.LatentContext[ParticleT, FilterLatentHistoryLength],
        Array,
        Array,
    ]: ...


def _select_ancestors(particles, ancestor_ix):
    return (
        particles
        if len(particles) == 0
        else jax.vmap(index_tree, in_axes=(None, 0, None))(
            particles,
            ancestor_ix,  # type: ignore[arg-type]
            0,
        )
    )


def multinomial_resample_from_log_weights(
    key,
    raw_log_weights,
    particles,
    num_resample,
):
    ancestor_ix = jrandom.categorical(key, raw_log_weights, shape=(num_resample,))
    resampled_particles = _select_ancestors(particles, ancestor_ix)
    resampled_log_w = -jnp.log(num_resample) * jnp.ones(
        (num_resample,), dtype=raw_log_weights.dtype
    )
    return resampled_particles, ancestor_ix, resampled_log_w


def systematic_resample_from_log_weights(
    key,
    raw_log_weights,
    particles,
    num_resample,
):
    """Systematically resample a weighted particle population."""

    weights = jax.nn.softmax(raw_log_weights)
    positions = (
        jrandom.uniform(key, (), minval=0.0, maxval=1.0) + jnp.arange(num_resample)
    ) / num_resample
    ancestor_ix = jnp.searchsorted(jnp.cumsum(weights), positions, side="right")
    ancestor_ix = jnp.minimum(ancestor_ix, weights.shape[0] - 1)
    resampled_particles = _select_ancestors(particles, ancestor_ix)
    resampled_log_w = -jnp.log(num_resample) * jnp.ones(
        (num_resample,), dtype=raw_log_weights.dtype
    )
    return resampled_particles, ancestor_ix, resampled_log_w


def no_resample(
    key,
    raw_log_weights,
    particles,
    num_resample,
):
    del key
    input_count = raw_log_weights.shape[0]
    if num_resample != input_count:
        raise ValueError(
            "no_resample requires num_resample to equal the input particle count, "
            f"got num_resample={num_resample} and input_count={input_count}"
        )
    return particles, jnp.arange(num_resample), raw_log_weights


def _ess_efficiency_from_log_weights(log_weights: Array) -> Array:
    """Return ESS divided by particle count, computed in log space."""

    log_weights = log_weights - jsp.special.logsumexp(log_weights)
    log_sum_weights_squared = jsp.special.logsumexp(2.0 * log_weights)
    count = log_weights.shape[0]
    return jnp.exp(-log_sum_weights_squared) / jnp.asarray(
        count, dtype=log_weights.dtype
    )


def conditional_resample(key, log_weights, particles, num_resample, threshold=0.5):
    ess_efficiency = _ess_efficiency_from_log_weights(log_weights)

    def resample_fn():
        return multinomial_resample_from_log_weights(
            key, log_weights, particles, num_resample
        )

    def no_resample_fn():
        return no_resample(key, log_weights, particles, num_resample)

    return jax.lax.cond(
        ess_efficiency < threshold,
        resample_fn,
        no_resample_fn,
    )
