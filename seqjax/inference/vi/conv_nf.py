import equinox as eqx
from jaxtyping import PRNGKeyArray, Array
from typing import Protocol, Callable,ClassVar
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
from paramax import Parameterize, AbstractUnwrappable
from paramax.utils import inv_softplus
from flowjax.bijections.masked_autoregressive import MaskedAutoregressive as FlowjaxMAF
from flowjax.distributions import (
    Normal as FlowjaxNormal,
    Transformed as FlowjaxTransformed,
)
from functools import partial
from flowjax.bijections.affine import TriangularAffine, solve_triangular
import equinox as eqx
import seqjax.model.typing as seqjtyping
import jax
import jaxtyping
import jax.numpy as jnp
import typing
import jax.random as jrandom
from .interface import LatentContext, LatentContextDims, AmortizedVariationalApproximation
from flowjax.utils import get_ravelled_pytree_constructor
from flowjax.bijections.masked_autoregressive import masked_autoregressive_mlp
import paramax

def _affine_with_min_scale(min_scale: float = 1e-6) -> Affine:
    scale = Parameterize(
        lambda x: softplus(x) + min_scale, 
        inv_softplus(1 - min_scale)
    )
    return eqx.tree_at(
        where=lambda aff: aff.scale, 
        pytree=Affine(), 
        replace=scale
    )

class ConditionalDiagonalAffine(AbstractBijection):
    """Condition-dependent diagonal affine transform over (sequence, target).

    For each time t,

        y[t] = loc[t] + scale[t] * x[t]

    where loc[t] and scale[t] are functions only of condition[t].
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    sequence_dim: int
    target_dim: int
    cond_dim: int
    min_scale: float
    net: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        sequence_dim: int,
        target_dim: int,
        cond_dim: int,
        nn_width: int,
        nn_depth: int,
        min_scale: float = 1e-6,
    ):
        self.sequence_dim = sequence_dim
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.shape = (sequence_dim, target_dim)
        self.cond_shape = (sequence_dim, cond_dim)
        self.min_scale = min_scale

        self.net = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=2 * target_dim,
            width_size=nn_width,
            depth=nn_depth,
            activation=jnn.relu,
            key=key,
        )

        # Initialise close to identity:
        # loc = 0, scale = 1.
        final = self.net.layers[-1]
        weight = jnp.zeros_like(final.weight)

        if final.bias is None:
            raise ValueError("Expected final affine head layer to have a bias.")

        bias = jnp.zeros_like(final.bias)
        bias = bias.at[target_dim:].set(inv_softplus(1.0 - min_scale))

        new_final = eqx.tree_at(lambda layer: layer.weight, final, weight)
        new_final = eqx.tree_at(lambda layer: layer.bias, new_final, bias)
        self.net = eqx.tree_at(lambda mlp: mlp.layers[-1], self.net, new_final)

    def _params(self, condition: Array) -> tuple[Array, Array]:
        if condition is None:
            raise ValueError("ConditionalDiagonalAffine requires condition.")

        params = eqx.filter_vmap(self.net)(condition)
        loc, raw_scale = jnp.split(params, 2, axis=-1)
        scale = jax.nn.softplus(raw_scale) + self.min_scale
        return loc, scale

    def transform_and_log_det(self, x, condition=None):
        loc, scale = self._params(condition)
        y = loc + scale * x
        log_det = jnp.sum(jnp.log(scale))
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        loc, scale = self._params(condition)
        x = (y - loc) / scale
        log_det = jnp.sum(jnp.log(scale))
        return x, -log_det
    
class AR1Accumulation(AbstractBijection):
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    unconstrained_ar: Array
    unconstrained_scale: Array
    loc: Array

    def __init__(
        self,
        *,
        sequence_dim: int,
        target_dim: int,
        init_ar: float = 0.9,
        init_scale: float = 1.0,
        init_loc: float = 0.0,
    ):
        self.shape = (sequence_dim, target_dim)

        self.unconstrained_ar = jnp.asarray(jnp.arctanh(init_ar))
        self.unconstrained_scale = inv_softplus(jnp.asarray(init_scale))
        self.loc = jnp.asarray(init_loc)

    @property
    def ar(self):
        return jnp.tanh(self.unconstrained_ar)

    @property
    def scale(self):
        return jax.nn.softplus(self.unconstrained_scale) + 1e-8

    def transform_and_log_det(self, eps, condition=None):
        ar = self.ar
        scale = self.scale
        loc = self.loc

        a = jnp.broadcast_to(ar, eps.shape)
        b = eps

        def compose(left, right):
            a_l, b_l = left
            a_r, b_r = right
            return a_r * a_l, b_r + a_r * b_l

        _, z = jax.lax.associative_scan(compose, (a, b), axis=0)

        x = loc + scale * z

        log_det = eps.size * jnp.log(scale)

        return x, log_det

    def inverse_and_log_det(self, x, condition=None):
        ar = self.ar
        scale = self.scale
        loc = self.loc

        z = (x - loc) / scale

        z_prev = jnp.concatenate(
            [jnp.zeros_like(z[:1]), z[:-1]],
            axis=0,
        )
        eps = z - ar * z_prev

        log_det = x.size * jnp.log(scale)

        return eps, -log_det

class ConditionalAR1Accumulation(AbstractBijection):
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]

    sequence_dim: int
    target_dim: int
    cond_dim: int

    min_scale: float
    max_abs_ar: float
    unconstrained_ar: Array
    net: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        sequence_dim: int,
        target_dim: int,
        cond_dim: int,
        nn_width: int,
        nn_depth: int,
        min_scale: float = 1e-8,
        max_abs_ar: float = 0.999,
        init_ar: float = 0.0,
        init_scale: float = 1.0,
        init_loc: float = 0.0,
    ):
        if abs(init_ar) >= max_abs_ar:
            raise ValueError(
                f"Expected abs(init_ar) < max_abs_ar, got "
                f"init_ar={init_ar}, max_abs_ar={max_abs_ar}."
            )
        if init_scale <= min_scale:
            raise ValueError(
                f"Expected init_scale > min_scale, got "
                f"init_scale={init_scale}, min_scale={min_scale}."
            )

        self.sequence_dim = sequence_dim
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.shape = (sequence_dim, target_dim)
        self.cond_shape = (sequence_dim, cond_dim)
        self.min_scale = min_scale
        self.max_abs_ar = max_abs_ar

        self.unconstrained_ar = jnp.arctanh(
            jnp.asarray(init_ar / max_abs_ar)
        )

        self.net = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=2 * target_dim,
            width_size=nn_width,
            depth=nn_depth,
            activation=jnn.relu,
            key=key,
        )

        # Initialise close to identity:
        #
        #   ar      ≈ init_ar
        #   scale_t ≈ init_scale
        #   loc_t   ≈ init_loc
        #
        # With the defaults, this gives x = eps at initialisation.
        final = self.net.layers[-1]
        if final.bias is None:
            raise ValueError("Expected final MLP layer to have a bias.")

        weight = jnp.zeros_like(final.weight)

        raw_scale_bias = inv_softplus(jnp.asarray(init_scale - min_scale))
        loc_bias = jnp.asarray(init_loc)

        bias = jnp.concatenate(
            [
                jnp.ones((target_dim,), dtype=final.bias.dtype)
                * raw_scale_bias.astype(final.bias.dtype),
                jnp.ones((target_dim,), dtype=final.bias.dtype)
                * loc_bias.astype(final.bias.dtype),
            ],
            axis=0,
        )

        new_final = eqx.tree_at(lambda layer: layer.weight, final, weight)
        new_final = eqx.tree_at(lambda layer: layer.bias, new_final, bias)
        self.net = eqx.tree_at(lambda mlp: mlp.layers[-1], self.net, new_final)

    @property
    def ar(self):
        return self.max_abs_ar * jnp.tanh(self.unconstrained_ar)

    def _params(self, condition: Array) -> tuple[Array, Array, Array]:
        if condition is None:
            raise ValueError("ConditionalLocScaleAR1Accumulation requires condition.")

        params = eqx.filter_vmap(self.net)(condition)
        raw_scale, loc = jnp.split(params, 2, axis=-1)

        ar = jnp.broadcast_to(self.ar, self.shape)
        scale = jax.nn.softplus(raw_scale) + self.min_scale

        return ar, scale, loc

    def transform_and_log_det(self, eps, condition=None):
        ar, scale, loc = self._params(condition)

        a = ar
        b = loc + scale * eps

        def compose(left, right):
            a_l, b_l = left
            a_r, b_r = right
            return a_r * a_l, b_r + a_r * b_l

        _, x = jax.lax.associative_scan(compose, (a, b), axis=0)

        log_det = jnp.sum(jnp.log(scale))
        return x, log_det

    def inverse_and_log_det(self, x, condition=None):
        ar, scale, loc = self._params(condition)

        x_prev = jnp.concatenate(
            [jnp.zeros_like(x[:1]), x[:-1]],
            axis=0,
        )

        eps = (x - ar * x_prev - loc) / scale

        log_det = jnp.sum(jnp.log(scale))
        return eps, -log_det

    
class Conditioner(Protocol):
    def __call__(self, x: Array, condition: Array) -> Array: ...


class _LocalConditioner(eqx.Module):
    """Shared local conditioner for parity coupling.

    A translated MLP is applied independently at each target dimension using local
    context windows extracted from the frozen parity and optional local conditions.
    """

    masked_autoregressive_mlp: Conditioner
    sequence_dim: int
    target_dim: int
    cond_dim: int
    radius: int
    target_len: int
    frozen_len: int
    update_even: bool
    _selector_ix: Array
    _input_available: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        sequence_dim: int,
        target_dim: int,
        cond_dim: int,
        radius: int,
        update_even: bool,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
    ):
        if radius < 0:
            raise ValueError(
                f"radius must be an integer >= 0; got {radius}.",
            )
        if sequence_dim < 2:
            raise ValueError("Local parity coupling requires dim >= 2.")

        self.sequence_dim = sequence_dim
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.radius = radius
        self.update_even = update_even
        self.target_len = (sequence_dim + int(update_even)) // 2
        self.frozen_len = self.sequence_dim - self.target_len

        selector_offset = 0 if update_even else 1
        self._selector_ix = (
            selector_offset
            + jnp.arange(self.target_len, dtype=jnp.int32)[:, None]
            + jnp.arange(2 * radius, dtype=jnp.int32)[None, :]
        )

        # ensure this is not differentiated
        self._input_available = jax.lax.stop_gradient(jnp.pad(
            jnp.ones((self.frozen_len,)),
            (radius, radius),
            mode="constant",
            constant_values=0.0,
        ))[self._selector_ix]

        # +1 for the available flag
        in_size = (target_dim + 1) * 2 * radius + cond_dim 

        if width_size < in_size:
            print(f"warning: conv NF conditioner width: {width_size}")
            print(f"total in size: {in_size} | {target_dim * 2 * radius} + {cond_dim}")
        
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

    def context_features(self, x_frozen: Array, target_conditions: Array) -> Array:
        """
        x_frozen: just the frozen locations
        """
        # pad 
        padded = jnp.pad(
            x_frozen, 
            ((self.radius, self.radius), (0, 0)), 
            mode="edge"
        )
        
        # select and flaten the trailing dim axis
        windows = padded[self._selector_ix].reshape(
            self.target_len, 
            2 * self.radius * self.target_dim
        )     

        return jnp.hstack((windows, self._input_available, target_conditions))

    def site_params(self, x_target: Array, context_features: Array) -> Array:
        # x_target: (target_dim,)
        # context_features: (target_dim * 2 * radius + cond_dim,)
        features = jnp.hstack((x_target, context_features))
        return self.masked_autoregressive_mlp(features)

    def __call__(self, x_target: Array, x_frozen: Array, target_conditions: Array) -> Array:
        ctx = self.context_features(x_frozen, target_conditions)
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
    _coord_to_rank: Array
    _target_ix: Array
    _frozen_ix: Array
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
        radius: int = 2,
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
            radius=radius,
            update_even=update_even,
            out_size=num_params,
            width_size=nn_width,
            depth=nn_depth,
            activation=nn_activation,
        )

        self._target_ix = jnp.arange(0 if self.update_even else 1, self.sequence_dim, 2, dtype=jnp.int32)
        self._frozen_ix = jnp.arange(1 if self.update_even else 0, self.sequence_dim, 2, jnp.int32)
        if self.update_even:
            self._coord_order = jnp.arange(self.target_dim, dtype=jnp.int32)
            self._coord_to_rank = jnp.arange(self.target_dim, dtype=jnp.int32)
        else:
            self._coord_order = jnp.arange(self.target_dim - 1, -1, -1, dtype=jnp.int32)
            self._coord_to_rank = jnp.arange(self.target_dim - 1, -1, -1, dtype=jnp.int32)

    def transform_and_log_det(self, x, condition=None):
        transformer = self._condition_to_transformer(x, condition)
        def transform_row(transformer_row, x_row):
            y_row, log_det_row = eqx.filter_vmap(
                lambda tr, xj: tr.transform_and_log_det(xj)
            )(transformer_row, x_row)
            return y_row, jnp.sum(log_det_row)


        y_target, log_det = eqx.filter_vmap(transform_row)(
            transformer,
            x[self._target_ix],
        )

        y = x.at[self._target_ix].set(y_target)
        return y, jnp.sum(log_det)


    def inverse_and_log_det(self, y, condition=None):
        # fixed temporal context for each target sequence site
        y_frozen = jnp.zeros_like(y).at[
            self.conditioner._frozen_ix
        ].set(y[self.conditioner._frozen_ix])
        ctx = self.conditioner.context_features(y_frozen, condition)
        y_target = y[self._target_indices]

        def invert_one_site(y_t, ctx_t):
            init_x_t = jnp.zeros_like(y_t)

            def scan_fn(x_t, j):
                params_flat = self.conditioner.site_params(x_t, ctx_t)
                params = params_flat.reshape(self.target_dim, -1)

                transformer_j = self.transformer_constructor(params[self._coord_to_rank[j]])
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
        x_target = x[self._target_ix]
        x_frozen = x[self._frozen_ix]
        target_conditions = condition[self._target_ix]
        transform_p = self.conditioner(x_target, x_frozen, target_conditions)
        transform_p = jnp.reshape(transform_p, (len(self._target_ix), x.shape[1], -1))
        # The masked network outputs are ordered by rank. For update_even=False,
        # ranks are reversed relative to natural coordinate order, so remap them
        # back to coordinate order before constructing per-coordinate transforms.
        transform_p = transform_p[:, self._coord_to_rank, :]
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
    radius: int = 3,
    add_conditional_affine: bool = False,
    add_ar_layer: bool = False,
    add_conditional_ar_layer: bool = False,
    invert: bool = False,
) -> Transformed:
    """Create a local odd/even coupling flow with translated shared conditioner weights.

    Each layer transforms one parity of dimensions using a shared local conditioner
    applied across sites. Conditioning variables are required and have shape
    ``(dim, cond_dim)``.
    """

    if add_conditional_ar_layer and add_ar_layer:
        raise ValueError("Cannot have both  conditional_ar_layer and ar_layer!")
    
    if transformer is None:
        transformer = _affine_with_min_scale(1e-8)

    loc = jnp.zeros((sequence_dim, target_dim))
    scale = jnp.ones((sequence_dim, target_dim))
    base_dist = FlowjaxNormal(loc, scale)

    num_keys = (
        flow_layers 
        + int(add_conditional_affine) 
        + int(add_ar_layer) 
        + int(add_conditional_ar_layer)
    )
    keys = jrandom.split(key, num_keys)

    layers = []
    key_ix = 0

    if add_conditional_affine:
        layers.append(
            ConditionalDiagonalAffine(
                keys[key_ix],
                sequence_dim=sequence_dim,
                target_dim=target_dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                min_scale=1e-6,
            )
        )
        key_ix += 1
    
    for layer_idx in range(flow_layers):
        layer_key = keys[key_ix]
        key_ix += 1

        layers.append(
            LocalParityCoupling(
                key=layer_key,
                transformer=transformer,
                sequence_dim=sequence_dim,
                target_dim=target_dim,
                update_even=layer_idx % 2 == 1,
                cond_dim=cond_dim,
                radius=radius,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation,
            )
        )

    if add_ar_layer:   
        final_layer = AR1Accumulation(
            sequence_dim=sequence_dim,
            target_dim=target_dim,
            init_ar=0.
        )
        layers.append(final_layer)

    if add_conditional_ar_layer:
        print("ADDED C AR ")
        final_layer = ConditionalAR1Accumulation(
            keys[-1],
            sequence_dim=sequence_dim,
            target_dim=target_dim,
            init_ar=0.9,
            cond_dim=cond_dim,
            nn_depth=2,
            nn_width=32,
        )
        layers.append(final_layer)

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
        latent_context_dims: LatentContextDims,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        flow_layers: int = 2,
        radius: int = 3,
        add_ar_layer: bool = False,
        add_conditional_ar_layer: bool = False,
        add_conditional_affine: bool = False,
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
            transformer = _affine_with_min_scale(1e-6)

        cond_input_dim = (
            latent_context_dims.parameter_context_dim
            + latent_context_dims.condition_context_dim
            + latent_context_dims.sequence_embedded_context_dim
        )
        self.distribution = local_parity_coupling_flow(
            key,
            sequence_dim=sample_length, 
            target_dim=target_struct_cls.flat_dim,
            flow_layers=flow_layers,
            transformer=transformer,
            cond_dim=cond_input_dim,
            radius=radius,
            nn_width=nn_width,
            nn_depth=nn_depth,
            add_ar_layer=add_ar_layer,
            add_conditional_affine=add_conditional_affine,
            add_conditional_ar_layer=add_conditional_ar_layer,
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
    ) -> tuple[TargetStructT, jaxtyping.Scalar, typing.Any]:
        
        cond = self._build_condition(condition)
        flat_sample, log_q = self.distribution.sample_and_log_prob(key, condition=cond)
        latent_sample = typing.cast(
            TargetStructT, self.target_struct_cls.unravel(flat_sample)
        )
        
        return latent_sample, log_q, state
