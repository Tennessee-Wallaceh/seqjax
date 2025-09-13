from _typeshed import Incomplete
from distrax._src.bijectors import (
    bijector as base,
    block as block,
    chain as chain,
    shift as shift,
    triangular_linear as triangular_linear,
    unconstrained_affine as unconstrained_affine,
)

Array: Incomplete

class LowerUpperTriangularAffine(chain.Chain):
    def __init__(self, matrix: Array, bias: Array) -> None: ...
    @property
    def lower(self) -> Array: ...
    @property
    def upper(self) -> Array: ...
    @property
    def matrix(self) -> Array: ...
    @property
    def bias(self) -> Array: ...
    def same_as(self, other: base.Bijector) -> bool: ...
