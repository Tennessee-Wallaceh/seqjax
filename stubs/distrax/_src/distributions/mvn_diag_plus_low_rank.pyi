from _typeshed import Incomplete
from distrax._src.bijectors.diag_linear import DiagLinear as DiagLinear
from distrax._src.bijectors.diag_plus_low_rank_linear import (
    DiagPlusLowRankLinear as DiagPlusLowRankLinear,
)
from distrax._src.distributions import distribution as distribution
from distrax._src.distributions.mvn_from_bijector import (
    MultivariateNormalFromBijector as MultivariateNormalFromBijector,
)
from distrax._src.utils import conversion as conversion

tfd: Incomplete
Array: Incomplete

class MultivariateNormalDiagPlusLowRank(MultivariateNormalFromBijector):
    equiv_tfp_cls: Incomplete
    def __init__(
        self,
        loc: Array | None = None,
        scale_diag: Array | None = None,
        scale_u_matrix: Array | None = None,
        scale_v_matrix: Array | None = None,
    ) -> None: ...
    @property
    def scale_diag(self) -> Array: ...
    @property
    def scale_u_matrix(self) -> Array: ...
    @property
    def scale_v_matrix(self) -> Array: ...
    def __getitem__(self, index) -> MultivariateNormalDiagPlusLowRank: ...
