import chex
from distrax._src.distributions import normal as normal

def setUpModule() -> None: ...

class NormalFloat64Test(chex.TestCase):
    def test_dtype(self, dtype) -> None: ...
