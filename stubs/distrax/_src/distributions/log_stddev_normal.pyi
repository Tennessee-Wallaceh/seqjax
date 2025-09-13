from _typeshed import Incomplete
from distrax._src.distributions import distribution as distribution, normal as normal
from distrax._src.utils import conversion as conversion

tfd: Incomplete
Array: Incomplete
Numeric: Incomplete

class LogStddevNormal(normal.Normal):
    def __init__(
        self, loc: Numeric, log_scale: Numeric, max_scale: float | None = None
    ) -> None: ...
    @property
    def log_scale(self) -> Array: ...
    def __getitem__(self, index) -> LogStddevNormal: ...
