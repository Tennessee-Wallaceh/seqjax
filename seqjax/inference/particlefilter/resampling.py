import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AncestorSample:
    """Indices and weights representing an ancestor distribution."""

    indices: Array
    log_weights: Array


class Resampler(typing.Protocol):
    """Return a weighted particle representation of the supplied log probabilities."""

    def __call__(
        self,
        key: PRNGKeyArray,
        raw_log_weights: Array,
        num_resample: int,
    ) -> AncestorSample: ...


def multinomial_resample_from_log_weights(
    key,
    raw_log_weights,
    num_resample,
):
    # jax.random.categorical takes unnormalised logits.
    ancestor_ix = jrandom.categorical(key, raw_log_weights, shape=(num_resample,))

    ancestor_sample_log_weight = jnp.full(
        (num_resample,),
        -jnp.log(num_resample),
        dtype=raw_log_weights.dtype,
    )
    return AncestorSample(ancestor_ix, ancestor_sample_log_weight)


def systematic_resample_from_log_weights(
    key,
    raw_log_weights,
    num_resample,
):
    """Systematically resample from normalized or unnormalized log weights."""

    log_weights = raw_log_weights - jsp.special.logsumexp(raw_log_weights)
    cumulative_weights = jnp.cumsum(jnp.exp(log_weights))
    # Force the final boundary to one so roundoff cannot produce an invalid index.
    cumulative_weights = cumulative_weights.at[-1].set(1.0)
    offset = jrandom.uniform(key, (), dtype=raw_log_weights.dtype) / num_resample
    positions = offset + jnp.arange(num_resample) / num_resample
    ancestor_ix = jnp.searchsorted(cumulative_weights, positions, side="right")

    ancestor_sample_log_weight = jnp.full(
        (num_resample,),
        -jnp.log(num_resample),
        dtype=raw_log_weights.dtype,
    )
    return AncestorSample(ancestor_ix, ancestor_sample_log_weight)


def no_resample(
    key,
    raw_log_weights,
    num_resample,
):
    num_particles = raw_log_weights.shape[0]
    if num_resample != num_particles:
        raise ValueError(
            "Identity resampling requires the requested output count to equal "
            f"the input particle count; received {num_resample} outputs for "
            f"{num_particles} particles"
        )
    return AncestorSample(jnp.arange(num_resample), raw_log_weights)


def _ess_efficiency_from_log_weights(log_weights: Array) -> Array:
    """
    ESS efficiency = ESS / N, with ESS = 1 / sum_i W_i^2 and W normalised.
    Computed stably in log space.
    """
    logW = log_weights - jsp.special.logsumexp(log_weights)  # log normalised weights
    log_sum_W2 = jsp.special.logsumexp(2.0 * logW)  # log(sum W^2)
    N = log_weights.shape[0]
    return jnp.exp(-log_sum_W2) / jnp.asarray(N, dtype=log_weights.dtype)


def conditional_resample(key, log_weights, num_resample, threshold=0.5):
    ess_efficiency = _ess_efficiency_from_log_weights(log_weights)

    def resample_fn():
        return multinomial_resample_from_log_weights(key, log_weights, num_resample)

    def no_resample_fn():
        return no_resample(key, log_weights, num_resample)

    return jax.lax.cond(
        ess_efficiency < threshold,
        resample_fn,
        no_resample_fn,
    )
