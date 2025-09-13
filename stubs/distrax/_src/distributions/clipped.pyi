from _typeshed import Incomplete
from distrax._src.distributions import (
    distribution as base_distribution,
    logistic as logistic,
    normal as normal,
)
from distrax._src.utils import conversion as conversion

Array: Incomplete
PRNGKey: Incomplete
Numeric: Incomplete
DistributionLike: Incomplete
EventT: Incomplete

class Clipped(base_distribution.Distribution):
    def __init__(
        self, distribution: DistributionLike, minimum: Numeric, maximum: Numeric
    ) -> None: ...
    def log_prob(self, value: EventT) -> Array: ...
    @property
    def minimum(self) -> Array: ...
    @property
    def maximum(self) -> Array: ...
    @property
    def distribution(self) -> DistributionLike: ...
    @property
    def event_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    def __getitem__(self, index) -> Clipped: ...

class ClippedNormal(Clipped):
    def __init__(
        self, loc: Numeric, scale: Numeric, minimum: Numeric, maximum: Numeric
    ) -> None: ...

class ClippedLogistic(Clipped):
    def __init__(
        self, loc: Numeric, scale: Numeric, minimum: Numeric, maximum: Numeric
    ) -> None: ...
