from _typeshed import Incomplete
from distrax._src.distributions import distribution as distribution

Array: Incomplete
DistributionLike: Incomplete

def importance_sampling_ratios(
    target_dist: DistributionLike, sampling_dist: DistributionLike, event: Array
) -> Array: ...
