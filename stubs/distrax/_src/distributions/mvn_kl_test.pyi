from absl.testing import parameterized
from distrax._src.distributions.mvn_diag import (
    MultivariateNormalDiag as MultivariateNormalDiag,
)
from distrax._src.distributions.mvn_diag_plus_low_rank import (
    MultivariateNormalDiagPlusLowRank as MultivariateNormalDiagPlusLowRank,
)
from distrax._src.distributions.mvn_full_covariance import (
    MultivariateNormalFullCovariance as MultivariateNormalFullCovariance,
)
from distrax._src.distributions.mvn_tri import (
    MultivariateNormalTri as MultivariateNormalTri,
)

class MultivariateNormalKLTest(parameterized.TestCase):
    def test_two_distributions(self, dist1_type, dist2_type) -> None: ...
