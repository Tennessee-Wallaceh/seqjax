from _typeshed import Incomplete
from distrax._src.distributions.mvn_diag_plus_low_rank import (
    MultivariateNormalDiagPlusLowRank as MultivariateNormalDiagPlusLowRank,
)
from distrax._src.utils import equivalence as equivalence

tfd: Incomplete

class MultivariateNormalDiagPlusLowRankTest(equivalence.EquivalenceTest):
    def setUp(self) -> None: ...
    def test_raises_on_wrong_inputs(self, dist_kwargs) -> None: ...
    def test_default_properties(self, dist_kwargs) -> None: ...
    def test_properties(
        self,
        batch_shape,
        loc_shape,
        scale_diag_shape,
        scale_u_matrix_shape,
        scale_v_matrix_shape,
    ) -> None: ...
    def test_sample_shape(
        self,
        sample_shape,
        loc_shape,
        scale_diag_shape,
        scale_u_matrix_shape,
        scale_v_matrix_shape,
    ): ...
    def test_sample_dtype(self, dtype) -> None: ...
    def test_log_prob(
        self,
        value_shape,
        loc_shape,
        scale_diag_shape,
        scale_u_matrix_shape,
        scale_v_matrix_shape,
    ) -> None: ...
    def test_method(
        self, loc_shape, scale_diag_shape, scale_u_matrix_shape, scale_v_matrix_shape
    ) -> None: ...
    def test_with_two_distributions(self, function_string, mode_string) -> None: ...
    def test_jittable(self) -> None: ...
    def test_slice(self, slice_) -> None: ...
    def test_slice_ellipsis(self) -> None: ...
