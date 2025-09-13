from _typeshed import Incomplete
from absl.testing import parameterized
from distrax._src.bijectors import bijector_from_tfp as bijector_from_tfp

tfb: Incomplete

class BijectorFromTFPTest(parameterized.TestCase):
    def setUp(self): ...
    def test_forward_methods_are_correct(
        self,
        tfp_bij_name,
        batch_shape_in,
        event_shape_in,
        batch_shape_out,
        event_shape_out,
    ) -> None: ...
    def test_inverse_methods_are_correct(
        self,
        tfp_bij_name,
        batch_shape_in,
        event_shape_in,
        batch_shape_out,
        event_shape_out,
    ) -> None: ...
    def test_composite_methods_are_consistent(
        self,
        tfp_bij_name,
        batch_shape_in,
        event_shape_in,
        batch_shape_out,
        event_shape_out,
    ) -> None: ...
    def test_works_with_tfp_caching(
        self,
        tfp_bij_name,
        batch_shape_in,
        event_shape_in,
        batch_shape_out,
        event_shape_out,
    ) -> None: ...
    def test_access_properties_tfp_bijector(self) -> None: ...
    def test_jittable(self): ...
