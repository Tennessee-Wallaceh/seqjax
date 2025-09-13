import jax.numpy as jnp
from _typeshed import Incomplete
from distrax._src.distributions import (
    categorical as categorical,
    distribution as distribution,
)
from typing import Any

Array: Incomplete

class EpsilonGreedy(categorical.Categorical):
    def __init__(
        self, preferences: Array, epsilon: float, dtype: jnp.dtype | type[Any] = ...
    ) -> None: ...
    @property
    def epsilon(self) -> float: ...
    @property
    def preferences(self) -> Array: ...
    def __getitem__(self, index) -> EpsilonGreedy: ...
