import jax.numpy as jnp
from _typeshed import Incomplete
from distrax._src.distributions import (
    categorical as categorical,
    distribution as distribution,
)
from distrax._src.utils import math as math
from typing import Any

tfd: Incomplete
Array: Incomplete
PRNGKey: Incomplete
EventT: Incomplete

class OneHotCategorical(categorical.Categorical):
    equiv_tfp_cls: Incomplete
    def __init__(
        self,
        logits: Array | None = None,
        probs: Array | None = None,
        dtype: jnp.dtype | type[Any] = ...,
    ) -> None: ...
    @property
    def event_shape(self) -> tuple[int, ...]: ...
    def log_prob(self, value: EventT) -> Array: ...
    def prob(self, value: EventT) -> Array: ...
    def mode(self) -> Array: ...
    def cdf(self, value: EventT) -> Array: ...
    def __getitem__(self, index) -> OneHotCategorical: ...
