from _typeshed import Incomplete
from distrax._src.bijectors.diag_linear import DiagLinear as DiagLinear
from distrax._src.bijectors.triangular_linear import (
    TriangularLinear as TriangularLinear,
)
from distrax._src.distributions import distribution as distribution
from distrax._src.distributions.mvn_from_bijector import (
    MultivariateNormalFromBijector as MultivariateNormalFromBijector,
)
from distrax._src.utils import conversion as conversion

tfd: Incomplete
Array: Incomplete

class MultivariateNormalTri(MultivariateNormalFromBijector):
    equiv_tfp_cls: Incomplete
    def __init__(
        self,
        loc: Array | None = None,
        scale_tri: Array | None = None,
        is_lower: bool = True,
    ) -> None: ...
    @property
    def scale_tri(self) -> Array: ...
    @property
    def is_lower(self) -> bool: ...
    def __getitem__(self, index) -> MultivariateNormalTri: ...
