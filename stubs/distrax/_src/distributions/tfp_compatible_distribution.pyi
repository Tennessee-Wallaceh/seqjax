from _typeshed import Incomplete
from distrax._src.distributions import distribution as distribution

tfd: Incomplete
Array: Incomplete
ArrayNumpy: Incomplete
Distribution: Incomplete
IntLike: Incomplete
PRNGKey: Incomplete
tangent_spaces: Incomplete
TangentSpace: Incomplete
EventT: Incomplete

def tfp_compatible_distribution(
    base_distribution: Distribution, name: str | None = None
) -> distribution.DistributionT: ...
