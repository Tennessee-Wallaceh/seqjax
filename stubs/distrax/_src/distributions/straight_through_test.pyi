from distrax._src.distributions import (
    one_hot_categorical as one_hot_categorical,
    straight_through as straight_through,
)
from distrax._src.utils import equivalence as equivalence, math as math

class StraightThroughTest(equivalence.EquivalenceTest):
    def setUp(self) -> None: ...
    def test_sample(self, dist_params, sample_shape): ...
