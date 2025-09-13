from absl.testing import parameterized
from distrax._src.bijectors import linear as linear

class MockLinear(linear.Linear):
    def forward_and_log_det(self, x) -> None: ...

class LinearTest(parameterized.TestCase):
    def test_properties(self, event_dims, batch_shape, dtype) -> None: ...
