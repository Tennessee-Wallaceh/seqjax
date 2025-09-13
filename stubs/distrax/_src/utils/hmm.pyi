import chex
from distrax._src.distributions import (
    categorical as categorical,
    distribution as distribution,
)
from distrax._src.utils import conversion as conversion, jittable as jittable

class HMM(jittable.Jittable):
    def __init__(
        self,
        init_dist: categorical.CategoricalLike,
        trans_dist: categorical.CategoricalLike,
        obs_dist: distribution.DistributionLike,
    ) -> None: ...
    @property
    def init_dist(self) -> categorical.CategoricalLike: ...
    @property
    def trans_dist(self) -> categorical.CategoricalLike: ...
    @property
    def obs_dist(self) -> distribution.DistributionLike: ...
    def sample(
        self, *, seed: chex.PRNGKey, seq_len: int
    ) -> tuple[chex.Array, chex.Array]: ...
    def forward(
        self, obs_seq: chex.Array, length: chex.Array | None = None
    ) -> tuple[float, chex.Array]: ...
    def backward(
        self, obs_seq: chex.Array, length: chex.Array | None = None
    ) -> chex.Array: ...
    def forward_backward(
        self, obs_seq: chex.Array, length: chex.Array | None = None
    ) -> tuple[chex.Array, chex.Array, chex.Array, float]: ...
    def viterbi(self, obs_seq: chex.Array) -> chex.Array: ...
