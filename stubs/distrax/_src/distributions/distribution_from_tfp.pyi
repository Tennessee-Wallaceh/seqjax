from _typeshed import Incomplete
from distrax._src.distributions import distribution as distribution

tfd: Incomplete
Array: Incomplete
PRNGKey: Incomplete
DistributionT: Incomplete
EventT: Incomplete

def distribution_from_tfp(tfp_distribution: tfd.Distribution) -> DistributionT: ...
