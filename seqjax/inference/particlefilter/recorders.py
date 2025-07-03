from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array

from seqjax.model.base import ParticleType


def current_particle_mean(
    extractor: Callable[[ParticleType], Array]
) -> Callable[[Array, tuple[ParticleType, ...]], Array]:
    """Return a recorder capturing the mean of ``extractor`` over particles."""

    def _recorder(weights: Array, particles: tuple[ParticleType, ...]) -> Array:
        current = extractor(particles[-1])
        return jnp.sum(current * jnp.expand_dims(weights, -1), axis=0)

    return _recorder
