from _typeshed import Incomplete
from absl.testing import parameterized
from distrax._src.distributions import distribution as distribution
from typing import Any, Callable

tfd: Incomplete
Array: Incomplete

def get_tfp_equiv(distrax_cls): ...

class EquivalenceTest(parameterized.TestCase):
    tfp_cls: Incomplete
    def setUp(self) -> None: ...
    def assertion_fn(self, **kwargs) -> Callable[[Any, Any], None]: ...
