from _typeshed import Incomplete
from distrax._src.bijectors import bijector as base, block as block
from distrax._src.utils import conversion as conversion
from typing import Any, Callable

Array: Incomplete
BijectorParams = Any

class SplitCoupling(base.Bijector):
    def __init__(
        self,
        split_index: int,
        event_ndims: int,
        conditioner: Callable[[Array], BijectorParams],
        bijector: Callable[[BijectorParams], base.BijectorLike],
        swap: bool = False,
        split_axis: int = -1,
    ) -> None: ...
    @property
    def bijector(self) -> Callable[[BijectorParams], base.BijectorLike]: ...
    @property
    def conditioner(self) -> Callable[[Array], BijectorParams]: ...
    @property
    def split_index(self) -> int: ...
    @property
    def swap(self) -> bool: ...
    @property
    def split_axis(self) -> int: ...
    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]: ...
    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]: ...
