from _typeshed import Incomplete
from distrax._src.distributions.distribution import DistributionLike as DistributionLike
from distrax._src.utils import conversion as conversion

tfd: Incomplete
PRNGKey: Incomplete

def estimate_kl_best_effort(
    distribution_a: DistributionLike,
    distribution_b: DistributionLike,
    rng_key: PRNGKey,
    num_samples: int,
    proposal_distribution: DistributionLike | None = None,
): ...
def mc_estimate_kl(
    distribution_a: DistributionLike,
    distribution_b: DistributionLike,
    rng_key: PRNGKey,
    num_samples: int,
    proposal_distribution: DistributionLike | None = None,
): ...
def mc_estimate_kl_with_reparameterized(
    distribution_a: DistributionLike,
    distribution_b: DistributionLike,
    rng_key: PRNGKey,
    num_samples: int,
): ...
def mc_estimate_mode(
    distribution: DistributionLike, rng_key: PRNGKey, num_samples: int
): ...
