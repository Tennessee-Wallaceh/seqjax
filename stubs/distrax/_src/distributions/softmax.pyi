import jax.numpy as jnp
from _typeshed import Incomplete
from distrax._src.distributions import (
    categorical as categorical,
    distribution as distribution,
)
from typing import Any

Array: Incomplete

class Softmax(categorical.Categorical):
    def __init__(
        self,
        logits: Array,
        temperature: float = 1.0,
        dtype: jnp.dtype | type[Any] = ...,
    ) -> None: ...
    @property
    def temperature(self) -> float: ...
    @property
    def unscaled_logits(self) -> Array: ...
    def __getitem__(self, index) -> Softmax: ...
