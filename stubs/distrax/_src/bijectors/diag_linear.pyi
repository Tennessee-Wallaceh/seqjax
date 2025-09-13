from _typeshed import Incomplete
from distrax._src.bijectors import (
    bijector as base,
    block as block,
    linear as linear,
    scalar_affine as scalar_affine,
)

Array: Incomplete

class DiagLinear(linear.Linear):
    forward: Incomplete
    forward_log_det_jacobian: Incomplete
    inverse: Incomplete
    inverse_log_det_jacobian: Incomplete
    inverse_and_log_det: Incomplete
    def __init__(self, diag: Array) -> None: ...
    @property
    def diag(self) -> Array: ...
    @property
    def matrix(self) -> Array: ...
    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]: ...
    def same_as(self, other: base.Bijector) -> bool: ...
