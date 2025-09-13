from _typeshed import Incomplete
from distrax._src.bijectors import bijector as base

Array: Incomplete

class RationalQuadraticSpline(base.Bijector):
    def __init__(
        self,
        params: Array,
        range_min: float,
        range_max: float,
        boundary_slopes: str = "unconstrained",
        min_bin_size: float = 0.0001,
        min_knot_slope: float = 0.0001,
    ) -> None: ...
    @property
    def num_bins(self) -> int: ...
    @property
    def knot_slopes(self) -> Array: ...
    @property
    def x_pos(self) -> Array: ...
    @property
    def y_pos(self) -> Array: ...
    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]: ...
    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]: ...
