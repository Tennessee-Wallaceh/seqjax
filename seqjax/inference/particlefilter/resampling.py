import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from seqjax.util import dynamic_index_pytree_in_dim as index_tree
import seqjax.model.typing as seqjtyping


class Resampler[
    ParticleT: seqjtyping.Latent,
](typing.Protocol):
    """
    Outputs:
    - resampled particles
    - ancestor indices
    - resampled log weights (the new log weights after resampling)

    - log normalizing constant adjustment from resampling
    - log weight increment adjustment terms

    The adjustment terms are necessary where the resampler changes the current distribution.
    """

    def __call__(
        self,
        key: PRNGKeyArray,
        log_weights: Array,
        particles: tuple[ParticleT, ...],
        num_resample: int,
    ) -> tuple[tuple[ParticleT, ...], Array, Array, typing.Any]: ...


def multinomial_resample_from_log_weights(
    key,
    raw_log_weights,
    particles,
    num_resample,
):
    # jax.random.categorical takes unnormalised logits.
    ancestor_ix = jrandom.categorical(key, raw_log_weights, shape=(num_resample,))

    resampled_particles = jax.vmap(
        index_tree,
        in_axes=[None, 0, None],
    )(
        particles,
        ancestor_ix,  # type: ignore[arg-type]
        0,
    )
    resampled_log_w = jnp.zeros((num_resample,), dtype=raw_log_weights.dtype)
    return resampled_particles, ancestor_ix, resampled_log_w, 0.0, 0.0


def no_resample(
    key,
    raw_log_weights,
    particles,
    num_resample,
):
    return particles, jnp.arange(num_resample), raw_log_weights, 0.0, 0.0


def _ess_efficiency_from_log_weights(log_weights: Array) -> Array:
    """
    ESS efficiency = ESS / N, with ESS = 1 / sum_i W_i^2 and W normalised.
    Computed stably in log space.
    """
    logW = log_weights - jsp.special.logsumexp(log_weights)  # log normalised weights
    log_sum_W2 = jsp.special.logsumexp(2.0 * logW)  # log(sum W^2)
    N = log_weights.shape[0]
    return jnp.exp(-log_sum_W2) / jnp.asarray(N, dtype=log_weights.dtype)


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
