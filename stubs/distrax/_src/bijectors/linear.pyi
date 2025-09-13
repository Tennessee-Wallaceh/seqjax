import abc
import jax.numpy as jnp
from _typeshed import Incomplete
from distrax._src.bijectors import bijector as base
from typing import Sequence

Array: Incomplete

class Linear(base.Bijector, metaclass=abc.ABCMeta):
    def __init__(
        self, event_dims: int, batch_shape: Sequence[int], dtype: jnp.dtype
    ) -> None: ...
    @property
    def matrix(self) -> Array: ...
    @property
    def event_dims(self) -> int: ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> jnp.dtype: ...
