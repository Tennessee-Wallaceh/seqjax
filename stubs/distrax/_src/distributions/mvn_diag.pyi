from _typeshed import Incomplete
from distrax._src.bijectors.diag_linear import DiagLinear as DiagLinear
from distrax._src.distributions import distribution as distribution
from distrax._src.distributions.mvn_from_bijector import (
    MultivariateNormalFromBijector as MultivariateNormalFromBijector,
)
from distrax._src.utils import conversion as conversion

tfd: Incomplete
Array: Incomplete
EventT: Incomplete

class MultivariateNormalDiag(MultivariateNormalFromBijector):
    equiv_tfp_cls: Incomplete
    def __init__(
        self, loc: Array | None = None, scale_diag: Array | None = None
    ) -> None: ...
    @property
    def scale_diag(self) -> Array: ...
    def cdf(self, value: EventT) -> Array: ...
    def log_cdf(self, value: EventT) -> Array: ...
    def __getitem__(self, index) -> MultivariateNormalDiag: ...
