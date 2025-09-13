from _typeshed import Incomplete
from absl.testing import parameterized
from distrax._src.distributions import (
    categorical as categorical,
    mvn_diag as mvn_diag,
    normal as normal,
)
from distrax._src.utils import hmm as hmm

tfd: Incomplete

class Function:
    def __init__(self, fn) -> None: ...
    def __call__(self, *args, **kwargs): ...

class HMMTest(parameterized.TestCase):
    def test_sample(self, length, num_states, obs_dist_name_and_params_fn) -> None: ...
    def test_forward_backward(
        self, length, num_states, obs_dist_name_and_params_fn
    ) -> None: ...
    def test_viterbi(self, length, num_states, obs_dist_name_and_params_fn) -> None: ...
    def test_viterbi_matches_specific_example(self) -> None: ...
