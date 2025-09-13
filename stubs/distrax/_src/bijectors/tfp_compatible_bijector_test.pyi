from _typeshed import Incomplete
from absl.testing import parameterized
from distrax._src.bijectors.block import Block as Block
from distrax._src.bijectors.chain import Chain as Chain
from distrax._src.bijectors.lambda_bijector import Lambda as Lambda
from distrax._src.bijectors.scalar_affine import ScalarAffine as ScalarAffine
from distrax._src.bijectors.tanh import Tanh as Tanh
from distrax._src.bijectors.tfp_compatible_bijector import (
    tfp_compatible_bijector as tfp_compatible_bijector,
)

tfb: Incomplete
tfd: Incomplete
RTOL: float

class TFPCompatibleBijectorTest(parameterized.TestCase):
    def test_transformed_distribution(
        self, dx_bijector_fn, tfp_bijector_fn, sample_shape
    ) -> None: ...
    def test_chain(self, dx_bijector_fn, tfb_bijector_fn, event) -> None: ...
    def test_invert(self, dx_bijector_fn, tfb_bijector_fn, event) -> None: ...
    def test_forward_and_inverse(
        self, dx_bijector_fn, tfp_bijector_fn, event
    ) -> None: ...
    def test_log_det_jacobian(self, dx_bijector_fn, tfp_bijector_fn, event) -> None: ...
    def test_batched_events(self, bij_fn, batch_shape) -> None: ...
    def test_with_different_event_ndims(self): ...
