from _typeshed import Incomplete
from absl.testing import parameterized
from distrax._src.distributions.categorical import Categorical as Categorical
from distrax._src.distributions.distribution import Distribution as Distribution
from distrax._src.distributions.distribution_from_tfp import (
    distribution_from_tfp as distribution_from_tfp,
)
from distrax._src.distributions.mvn_diag import (
    MultivariateNormalDiag as MultivariateNormalDiag,
)
from distrax._src.distributions.normal import Normal as Normal
from distrax._src.distributions.transformed import Transformed as Transformed

tfb: Incomplete
tfd: Incomplete

class DistributionFromTfpNormal(parameterized.TestCase):
    base_dist: Incomplete
    values: Incomplete
    distrax_second_dist: Incomplete
    tfp_second_dist: Incomplete
    def setUp(self) -> None: ...
    def assertion_fn(self, rtol): ...
    @property
    def wrapped_dist(self): ...
    def test_event_shape(self) -> None: ...
    def test_batch_shape(self) -> None: ...
    def test_sample_dtype(self) -> None: ...
    def test_sample(self): ...
    def test_method(self, method) -> None: ...
    def test_method_with_value(self, method) -> None: ...
    def test_sample_and_log_prob(self): ...
    def test_with_two_distributions(self, method) -> None: ...

class DistributionFromTfpMvnNormal(DistributionFromTfpNormal):
    base_dist: Incomplete
    values: Incomplete
    distrax_second_dist: Incomplete
    tfp_second_dist: Incomplete
    def setUp(self) -> None: ...
    def test_slice(self, slice_) -> None: ...

class DistributionFromTfpCategorical(DistributionFromTfpNormal):
    base_dist: Incomplete
    values: Incomplete
    distrax_second_dist: Incomplete
    tfp_second_dist: Incomplete
    def setUp(self) -> None: ...
    def test_slice(self, slice_) -> None: ...
    def test_slice_ellipsis(self) -> None: ...

class DistributionFromTfpTransformed(DistributionFromTfpNormal):
    base_dist: Incomplete
    values: Incomplete
    distrax_second_dist: Incomplete
    tfp_second_dist: Incomplete
    def setUp(self) -> None: ...
