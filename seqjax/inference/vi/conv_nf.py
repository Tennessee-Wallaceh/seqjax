import equinox as eqx
from jaxtyping import PRNGKeyArray, Array
from typing import Protocol, Callable
import jax.nn as jnn
import jax.numpy as jnp
from flowjax.bijections import (
    AbstractBijection,
    Chain,
    Invert,
    Affine,
)
from jax.nn import softplus
from flowjax.distributions import AbstractDistribution, Transformed
from paramax import Parameterize
from paramax.utils import inv_softplus
from flowjax.bijections.masked_autoregressive import MaskedAutoregressive as FlowjaxMAF
from flowjax.distributions import (
    Normal as FlowjaxNormal,
    Transformed as FlowjaxTransformed,
)
import equinox as eqx
import seqjax.model.typing as seqjtyping
import jax
import jaxtyping
import jax.numpy as jnp
import typing
import jax.random as jrandom
from .interface import Embedder, LatentContext, AmortizedVariationalApproximation
from flowjax.utils import get_ravelled_pytree_constructor
from flowjax.bijections.masked_autoregressive import masked_autoregressive_mlp
import paramax

def _affine_with_min_scale(min_scale: float = 1e-2) -> Affine:
    scale = Parameterize(lambda x: softplus(x) + min_scale, inv_softplus(1 - min_scale))
    return eqx.tree_at(where=lambda aff: aff.scale, pytree=Affine(), replace=scale)


class Conditioner(Protocol):
    def __call__(self, x: Array) -> Array: ...


class _LocalConditioner(eqx.Module):
    """Shared local conditioner for parity coupling.

    A translated MLP is applied independently at each target dimension using local
    context windows extracted from the frozen parity and optional local conditions.
    """

    masked_autoregressive_mlp: Conditioner
    sequence_dim: int
    target_dim: int
    cond_dim: int
    kernel_size: int
    update_even: bool
    _target_ix: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        sequence_dim: int,
        target_dim: int,
        cond_dim: int,
        kernel_size: int,
        update_even: bool,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
    ):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be an odd integer >= 3; got {kernel_size}.",
            )
        if sequence_dim < 2:
            raise ValueError("Local parity coupling requires dim >= 2.")

        self.sequence_dim = sequence_dim
        self.target_dim = target_dim
        self._target_ix = jnp.arange(0 if update_even else 1, self.sequence_dim, 2, jnp.int32)
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.update_even = update_even
        in_size = target_dim * (kernel_size - 1) + cond_dim

        if width_size < in_size:
            print(f"warning: conv NF conditioner width: {width_size}")
            print(f"total in size: {in_size} | {target_dim * (kernel_size - 1)}, {cond_dim}")
        
        # we give conditioning variables rank -1 (no masking of edges to output)
        if update_even:
            in_ranks = jnp.hstack((jnp.arange(target_dim), -jnp.ones(in_size, int)))
        else:
            in_ranks = jnp.hstack((jnp.arange(target_dim)[::-1], -jnp.ones(in_size, int)))

        # If dim=1, hidden ranks all -1 -> all outputs only depend on condition
        hidden_ranks = (jnp.arange(width_size) % target_dim) - 1
        out_ranks = jnp.repeat(jnp.arange(target_dim), out_size)

        self.masked_autoregressive_mlp = masked_autoregressive_mlp(
            in_ranks,
            hidden_ranks,
            out_ranks,
            depth=depth,
            activation=activation,
            key=key,
        )

    def context_features(self, x: Array, condition: Array) -> Array:
        pad = self.kernel_size - 2
        offsets = jnp.arange(-(self.kernel_size - 2), self.kernel_size, 2)
        padded = jnp.pad(x, ((pad, pad), (0, 0)), mode="edge")
        windows = padded[
            self._target_ix[:, None] + pad + offsets
        ].reshape(len(self._target_ix), -1)
        cond_features = condition[self._target_ix]
        return jnp.hstack((windows, cond_features))

    def site_params(self, x_target: Array, context_features: Array) -> Array:
        # x_target: (target_dim,)
        # context_features: (target_dim * (kernel_size - 1) + cond_dim,)
        features = jnp.hstack((x_target, context_features))
        return self.masked_autoregressive_mlp(features)

    def __call__(self, x: Array, condition: Array) -> Array:
        x_target = x[self._target_ix]
        ctx = self.context_features(x, condition)
        return eqx.filter_vmap(self.site_params)(x_target, ctx)
    
class LocalParityCoupling(AbstractBijection):
    """Odd/even local coupling layer with shared translated conditioner weights.

    The transformed subset is selected by parity. For each transformed site, a shared
    conditioner network is applied to a local receptive field extracted from the frozen
    parity together with a required dimension-local condition.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    sequence_dim: int
    target_dim: int
    update_even: bool
    _coord_order: Array
    _target_indices: Array
    transformer_constructor: Callable
    conditioner: _LocalConditioner

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        transformer: AbstractBijection,
        sequence_dim: int,
        target_dim: int,
        update_even: bool,
        cond_dim: int,
        kernel_size: int = 3,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ):
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

        self.transformer_constructor = constructor
        self.shape = (sequence_dim, target_dim)
        self.cond_shape = (sequence_dim, cond_dim)
        self.sequence_dim = sequence_dim
        self.target_dim = target_dim
        self.update_even = update_even
        self.conditioner = _LocalConditioner(
            key,
            sequence_dim=sequence_dim,
            target_dim=target_dim,
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            update_even=update_even,
            out_size=num_params,
            width_size=nn_width,
            depth=nn_depth,
            activation=nn_activation,
        )

        self._target_indices = jnp.arange(0 if self.update_even else 1, self.sequence_dim, 2, dtype=jnp.int32)
        if self.update_even:
            self._coord_order = jnp.arange(self.target_dim, dtype=jnp.int32)
        else:
            self._coord_order = jnp.arange(self.target_dim - 1, -1, -1, dtype=jnp.int32)

    def transform_and_log_det(self, x, condition=None):
        transformer = self._condition_to_transformer(x, condition)
        def transform_row(transformer_row, x_row):
            y_row, log_det_row = eqx.filter_vmap(
                lambda tr, xj: tr.transform_and_log_det(xj)
            )(transformer_row, x_row)
            return y_row, jnp.sum(log_det_row)

        y_target, log_det = eqx.filter_vmap(transform_row)(
            transformer,
            x[self._target_indices],
        )

        y = x.at[self._target_indices].set(y_target)
        return y, jnp.sum(log_det)


    def inverse_and_log_det(self, y, condition=None):
        # fixed temporal context for each target sequence site
        ctx = self.conditioner.context_features(y, condition)
        y_target = y[self._target_indices]

        def invert_one_site(y_t, ctx_t):
            init_x_t = jnp.zeros_like(y_t)

            def scan_fn(x_t, j):
                params_flat = self.conditioner.site_params(x_t, ctx_t)
                params = params_flat.reshape(self.target_dim, -1)

                transformer_j = self.transformer_constructor(params[j])
                x_j = transformer_j.inverse(y_t[j])

                x_t = x_t.at[j].set(x_j)
                return x_t, None

            x_t, _ = jax.lax.scan(scan_fn, init_x_t, self._coord_order)
            return x_t

        x_target = eqx.filter_vmap(invert_one_site)(y_target, ctx)
        x = y.at[self._target_indices].set(x_target)

        log_det = self.transform_and_log_det(x, condition)[1]
        return x, -log_det

    def _condition_to_transformer(self, x: Array, condition: Array):
        transform_p = self.conditioner(x, condition)
        transform_p = jnp.reshape(transform_p, (len(self._target_indices), x.shape[1], -1))
        transformer = eqx.filter_vmap(eqx.filter_vmap(self.transformer_constructor))(transform_p)
        return transformer

def local_parity_coupling_flow(
    key: PRNGKeyArray,
    *,
    sequence_dim,
    target_dim,
    cond_dim: int,
    transformer: AbstractBijection | None = None,
    flow_layers: int = 2,
    nn_width: int = 50,
    nn_depth: int = 2,
    nn_activation: Callable = jnn.relu,
    kernel_size: int = 3,
    invert: bool = False,
) -> Transformed:
    """Create a local odd/even coupling flow with translated shared conditioner weights.

    Each layer transforms one parity of dimensions using a shared local conditioner
    applied across sites. Conditioning variables are required and have shape
    ``(dim, cond_dim)``.
    """
    if transformer is None:
        transformer = _affine_with_min_scale()

    loc = jnp.zeros((sequence_dim, target_dim))
    scale = jnp.ones((sequence_dim, target_dim))
    base_dist = FlowjaxNormal(loc, scale)

    keys = jrandom.split(key, flow_layers)
    layers = [
        LocalParityCoupling(
            key=layer_key,
            transformer=transformer,
            sequence_dim=sequence_dim,
            target_dim=target_dim,
            update_even=layer_idx % 2 == 1,
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        for layer_idx, layer_key in enumerate(keys)
    ]
    bijection = Chain(layers).merge_chains()
    bijection = Invert(bijection) if invert else bijection
    return Transformed(base_dist, bijection)


class AmortizedConvCoupling[
    TargetStructT: seqjtyping.Packable,
](AmortizedVariationalApproximation[TargetStructT]):
    """Conditional masked autoregressive flow over buffered latent paths."""
    distribution: FlowjaxTransformed

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        sample_length: int,
        embedder: Embedder,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        flow_layers: int = 2,
        kernel_size: int = 3,
        transformer: AbstractBijection | None = None,
    ) -> None:
        
        shape = (sample_length, target_struct_cls.flat_dim)
        super().__init__(
            target_struct_cls,
            shape=shape,
            sample_length=sample_length,
        )
        self.target_struct_cls = target_struct_cls

        if transformer is None:
            transformer = _affine_with_min_scale(1e-4)

        cond_input_dim = (
            embedder.parameter_context_dim
            + embedder.condition_context_dim
            + embedder.sequence_embedded_context_dim
        )

        self.distribution = local_parity_coupling_flow(
            key,
            sequence_dim=sample_length, 
            target_dim=target_struct_cls.flat_dim,
            flow_layers=flow_layers,
            transformer=transformer,
            cond_dim=cond_input_dim,
            kernel_size=kernel_size,
            nn_width=nn_width,
            nn_depth=nn_depth,
            invert=False
        )
        

    def _build_condition(
        self,
        condition: LatentContext,
    ) -> jaxtyping.Array:
        # Have to broadcast context info down the sequence axis

        parameter_context = jnp.broadcast_to(
            jnp.expand_dims(condition.parameter_context.ravel().flatten(), axis=-2),
            (
                *condition.parameter_context.batch_shape,  
                self.sample_length, 
                condition.parameter_context.flat_dim
            ),
        )
        condition_context = jnp.broadcast_to(
            jnp.expand_dims(condition.condition_context.ravel().flatten(), axis=-2),
            (
                *condition.condition_context.batch_shape, 
                self.sample_length, 
                condition.condition_context.flat_dim
            ),
        )
        return jnp.concatenate(
            [
                parameter_context, 
                condition_context, 
                condition.sequence_embedded_context, 
            ], axis=-1
        )

    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: LatentContext,
        state: typing.Any = None,
        *,
        inference: bool = False,
    ) -> tuple[TargetStructT, jaxtyping.Scalar, typing.Any]:
        
        cond = self._build_condition(condition)
        flat_sample, log_q = self.distribution.sample_and_log_prob(key, condition=cond)
        latent_sample = typing.cast(
            TargetStructT, self.target_struct_cls.unravel(flat_sample)
        )
        return latent_sample, log_q, state
