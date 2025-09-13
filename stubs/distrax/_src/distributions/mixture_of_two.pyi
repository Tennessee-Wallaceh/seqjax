from _typeshed import Incomplete
from distrax._src.distributions import distribution as base_distribution
from distrax._src.utils import conversion as conversion

tfd: Incomplete
Array: Incomplete
Numeric: Incomplete
PRNGKey: Incomplete
DistributionLike: Incomplete
EventT: Incomplete

class MixtureOfTwo(base_distribution.Distribution):
    def __init__(
        self,
        prob_a: Numeric,
        component_a: DistributionLike,
        component_b: DistributionLike,
    ) -> None: ...
    def log_prob(self, value: EventT) -> Array: ...
    @property
    def event_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def prob_a(self) -> Numeric: ...
    @property
    def prob_b(self) -> Numeric: ...
    def __getitem__(self, index) -> MixtureOfTwo: ...
