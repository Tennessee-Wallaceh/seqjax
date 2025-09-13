from _typeshed import Incomplete
from distrax._src.bijectors import (
    bijector as bijector,
    bijector_from_tfp as bijector_from_tfp,
    lambda_bijector as lambda_bijector,
    sigmoid as sigmoid,
    tanh as tanh,
    tfp_compatible_bijector as tfp_compatible_bijector,
)
from distrax._src.distributions import (
    distribution as distribution,
    distribution_from_tfp as distribution_from_tfp,
    tfp_compatible_distribution as tfp_compatible_distribution,
)

tfb: Incomplete
tfd: Incomplete
Array: Incomplete
Numeric: Incomplete
BijectorLike: Incomplete
DistributionLike: Incomplete

def to_tfp(
    obj: bijector.Bijector
    | tfb.Bijector
    | distribution.Distribution
    | tfd.Distribution,
    name: str | None = None,
): ...
def as_bijector(obj: BijectorLike) -> bijector.BijectorT: ...
def as_distribution(obj: DistributionLike) -> distribution.DistributionT: ...
def as_float_array(x: Numeric) -> Array: ...
