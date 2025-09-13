from _typeshed import Incomplete
from distrax._src.distributions import distribution as distribution
from distrax._src.distributions.mvn_tri import (
    MultivariateNormalTri as MultivariateNormalTri,
)
from distrax._src.utils import conversion as conversion

tfd: Incomplete
Array: Incomplete

class MultivariateNormalFullCovariance(MultivariateNormalTri):
    equiv_tfp_cls: Incomplete
    def __init__(
        self, loc: Array | None = None, covariance_matrix: Array | None = None
    ) -> None: ...
    @property
    def covariance_matrix(self) -> Array: ...
    def covariance(self) -> Array: ...
    def variance(self) -> Array: ...
    def __getitem__(self, index) -> MultivariateNormalFullCovariance: ...
