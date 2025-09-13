import chex
from distrax._src.bijectors import (
    rational_quadratic_spline as rational_quadratic_spline,
)

def setUpModule() -> None: ...

class RationalQuadraticSplineFloat64Test(chex.TestCase):
    def test_dtypes(self, dtypes, boundary_slopes) -> None: ...
