from flowjax.bijections import Affine, AbstractBijection
from flowjax.bijections.masked_autoregressive import MaskedAutoregressive as FlowjaxMAF
from flowjax.distributions import (
    Normal as FlowjaxNormal,
    Transformed as FlowjaxTransformed,
)
from flowjax.flows import masked_autoregressive_flow
import equinox as eqx
import seqjax.model.typing as seqjtyping
from seqjax.model.base import SequentialModel
from seqjax.model import simulate, evaluate

import jaxtyping
import jax
import jax.numpy as jnp
import jax.random as jrandom
import typing

from .api import Embedder, LatentContext, VariationalApproximationFactory, UnconditionalVariationalApproximation, AmortizedVariationalApproximation


def _ensure_legacy_key(key: jaxtyping.PRNGKeyArray) -> jaxtyping.PRNGKeyArray:
    """Convert legacy uint32 keys to typed JAX keys for FlowJax."""
    return jrandom.wrap_key_data(jnp.asarray(key, dtype=jnp.uint32))

class MaskedAutoregressiveFlow[
    TargetStructT: seqjtyping.Packable,
](UnconditionalVariationalApproximation[TargetStructT]):
    """Masked autoregressive flow over the flattened parameter space."""

    target_struct_cls: type[TargetStructT]
    base_distribution: FlowjaxNormal
    flow: FlowjaxMAF
    distribution: FlowjaxTransformed

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        super().__init__(target_struct_cls, shape=(target_struct_cls.flat_dim,))
        self.target_struct_cls = target_struct_cls
        dim = target_struct_cls.flat_dim

        loc = jnp.broadcast_to(jnp.asarray(base_loc), (dim,))
        scale = jnp.broadcast_to(jnp.asarray(base_scale), (dim,))
        self.base_distribution = FlowjaxNormal(loc, scale)

        if transformer is None:
            transformer = Affine()

        self.flow = FlowjaxMAF(
            key,
            transformer=transformer,
            dim=dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
        distribution = typing.cast(
            FlowjaxTransformed,
            FlowjaxTransformed(self.base_distribution, self.flow),  # type: ignore[arg-type, call-arg]
        )
        self.distribution = distribution

    def sample_and_log_prob(self, key, condition=None):
        flat_sample = self.distribution.sample(key)
        log_q = self.distribution.log_prob(flat_sample)
        return self.target_struct_cls.unravel(flat_sample), log_q


class MaskedAutoregressiveFlowFactory[TargetStructT: seqjtyping.Packable](
    VariationalApproximationFactory[TargetStructT, None]
):
    def __init__(
        self,
        *,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        self._key = key
        self._nn_width = nn_width
        self._nn_depth = nn_depth
        self._base_loc = base_loc
        self._base_scale = base_scale
        self._transformer = transformer

    def __call__(
        self, target_struct_cls: type[TargetStructT]
    ) -> MaskedAutoregressiveFlow[TargetStructT]:
        return MaskedAutoregressiveFlow(
            target_struct_cls,
            key=self._key,
            nn_width=self._nn_width,
            nn_depth=self._nn_depth,
            base_loc=self._base_loc,
            base_scale=self._base_scale,
            transformer=self._transformer,
        )


class AmortizedMAF[
    TargetStructT: seqjtyping.Packable,
](AmortizedVariationalApproximation[TargetStructT]):
    """Conditional masked autoregressive flow over buffered latent paths."""
    distribution: FlowjaxTransformed
    _flat_sample_dim: int = eqx.field(static=True)

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        sample_length: int,
        embedder: Embedder,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        flow_layers: int = 1,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        
        shape = (sample_length, target_struct_cls.flat_dim)
        super().__init__(
            target_struct_cls,
            shape=shape,
            sample_length=sample_length,
        )
        self.target_struct_cls = target_struct_cls
        flat_sample_dim = sample_length * target_struct_cls.flat_dim
        self._flat_sample_dim = flat_sample_dim

        if transformer is None:
            transformer = Affine()

        loc = jnp.broadcast_to(jnp.asarray(base_loc), (flat_sample_dim,))
        scale = jnp.broadcast_to(jnp.asarray(base_scale), (flat_sample_dim,))

        cond_input_dim = (
            embedder.parameter_context_dim
            + embedder.condition_context_dim
            + embedder.embedded_context_dim
        )

        self.distribution = masked_autoregressive_flow(
            key,
            base_dist=FlowjaxNormal(loc, scale),
            flow_layers=flow_layers,
            transformer=transformer,
            cond_dim=cond_input_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            invert=False
        )
        

    def _build_condition(
        self,
        condition: LatentContext,
    ) -> jaxtyping.Array:
        return jnp.concatenate(
            [
                condition.parameter_context.ravel().flatten(), 
                condition.condition_context.ravel().flatten(), 
                condition.embedded_context, 
            ], axis=0
        )

    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: LatentContext,
    ) -> tuple[TargetStructT, jaxtyping.Scalar]:
        
        cond = self._build_condition(condition)
        flat_sample, log_q = self.distribution.sample_and_log_prob(key, condition=cond)
        reshaped_sample = jnp.reshape(flat_sample, self.shape)
        latent_sample = typing.cast(
            TargetStructT, self.target_struct_cls.unravel(reshaped_sample)
        )
        return latent_sample, log_q