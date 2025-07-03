from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Scalar


def compute_esse_from_log_weights(log_weights: Array) -> Scalar:
    """Return the effective sample size efficiency for ``log_weights``."""
    log_sum_w = jax.scipy.special.logsumexp(log_weights)
    log_sum_w2 = jax.scipy.special.logsumexp(2 * log_weights)
    ess = jnp.exp(2 * log_sum_w - log_sum_w2)
    return ess / log_weights.shape[0]
