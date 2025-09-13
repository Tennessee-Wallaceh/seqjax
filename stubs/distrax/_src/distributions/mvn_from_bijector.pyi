from _typeshed import Incomplete
from distrax._src.bijectors import (
    block as block,
    chain as chain,
    diag_linear as diag_linear,
    linear as linear,
    shift as shift,
)
from distrax._src.distributions import (
    independent as independent,
    normal as normal,
    transformed as transformed,
)

tfd: Incomplete
Array: Incomplete

class MultivariateNormalFromBijector(transformed.Transformed):
    def __init__(self, loc: Array, scale: linear.Linear) -> None: ...
    @property
    def scale(self) -> linear.Linear: ...
    @property
    def loc(self) -> Array: ...
    def mean(self) -> Array: ...
    def median(self) -> Array: ...
    def mode(self) -> Array: ...
    def covariance(self) -> Array: ...
    def variance(self) -> Array: ...
    def stddev(self) -> Array: ...

MultivariateNormalLike: Incomplete
