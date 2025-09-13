import chex
from distrax._src.distributions import uniform as uniform

def setUpModule() -> None: ...

class UniformFloat64Test(chex.TestCase):
    def test_dtype(self, dtype) -> None: ...
