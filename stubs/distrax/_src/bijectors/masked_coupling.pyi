from _typeshed import Incomplete
from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion as conversion, math as math
from typing import Any, Callable

Array: Incomplete
BijectorParams = Any

class MaskedCoupling(base.Bijector):
    def __init__(
        self,
        mask: Array,
        conditioner: Callable[[Array], BijectorParams],
        bijector: Callable[[BijectorParams], base.BijectorLike],
        event_ndims: int | None = None,
        inner_event_ndims: int = 0,
    ) -> None: ...
    @property
    def bijector(self) -> Callable[[BijectorParams], base.BijectorLike]: ...
    @property
    def conditioner(self) -> Callable[[Array], BijectorParams]: ...
    @property
    def mask(self) -> Array: ...
    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]: ...
    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]: ...
