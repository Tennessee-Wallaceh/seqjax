from _typeshed import Incomplete
from distrax._src.distributions import (
    categorical as categorical,
    distribution as distribution,
)
from distrax._src.utils import conversion as conversion

tfd: Incomplete
Array: Incomplete
PRNGKey: Incomplete
DistributionLike: Incomplete
CategoricalLike: Incomplete
EventT: Incomplete

class MixtureSameFamily(distribution.Distribution):
    equiv_tfp_cls: Incomplete
    def __init__(
        self,
        mixture_distribution: CategoricalLike,
        components_distribution: DistributionLike,
    ) -> None: ...
    @property
    def components_distribution(self): ...
    @property
    def mixture_distribution(self): ...
    @property
    def event_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    def log_prob(self, value: EventT) -> Array: ...
    def mean(self) -> Array: ...
    def variance(self) -> Array: ...
    def __getitem__(self, index) -> MixtureSameFamily: ...
