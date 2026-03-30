from typing import Protocol, Type
import typing

from seqjax.model.interface import BayesianSequentialModelProtocol
from seqjax.model import typing as seqjtyping
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .interface import LatentContextDims, LatentContext, AmortizedVariationalApproximation, UnconditionalVariationalApproximation


_LOG_2PI = jnp.log(2.0 * jnp.pi)


def xavier_init(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    fan_in, fan_out = shape[1], shape[0]
    std = jnp.sqrt(2.0 / (fan_in + fan_out))
    return jrandom.normal(key, shape) * std


def make_linear(in_dim: int, out_dim: int, key: PRNGKeyArray) -> eqx.nn.Linear:
    layer = eqx.nn.Linear(in_dim, out_dim, key=key)
    wkey, _ = jrandom.split(key)
    new_weight = xavier_init(wkey, layer.weight.shape)
    assert layer.bias is not None
    new_bias = jnp.zeros(layer.bias.shape, dtype=layer.bias.dtype)

    layer = eqx.tree_at(lambda leaf: leaf.weight, layer, new_weight)
    layer = eqx.tree_at(lambda leaf: leaf.bias, layer, new_bias)
    return layer


class ResidualBlock(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, width: int, key: PRNGKeyArray):
        k1, k2 = jrandom.split(key, 2)
        self.linear1 = make_linear(width, width, k1)
        self.linear2 = make_linear(width, width, k2)

    def __call__(self, x: Array) -> Array:
        out = self.linear1(x)
        out = jax.nn.relu(out)
        out = self.linear2(out)
        return jax.nn.relu(out + x)


class ResNetMLP(eqx.Module):
    input_proj: eqx.nn.Linear
    blocks: list[ResidualBlock]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        width: int,
        out_size: int,
        depth: int,
        *,
        use_batchnorm: bool,
        key: PRNGKeyArray,
    ) -> None:
        keys = jrandom.split(key, depth + 2)
        self.input_proj = make_linear(in_size, width, keys[0])
        self.blocks = [ResidualBlock(width, k) for k in keys[1:-1]]
        self.output_proj = make_linear(width, out_size, keys[-1])

    def __call__(self, x: Array) -> Array:
        x = jax.nn.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class AmortizerMLP(eqx.Module):
    """Alternative MLP for amortisation."""

    proj_x: eqx.nn.Linear
    proj_theta: eqx.nn.Linear
    proj_context: eqx.nn.Linear
    proj_missing: eqx.nn.Linear
    mlp: eqx.nn.MLP

    def __init__(
        self,
        input_dims: tuple[int, int, int, int],
        hidden_dim: int,
        mlp_output_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2, k3, k4, k5 = jrandom.split(key, 5)
        self.proj_x = eqx.nn.Linear(input_dims[0], hidden_dim, key=k1)
        self.proj_theta = eqx.nn.Linear(input_dims[1], hidden_dim, key=k2)
        self.proj_context = eqx.nn.Linear(input_dims[2], hidden_dim, key=k3)
        self.proj_missing = eqx.nn.Linear(input_dims[3], hidden_dim, key=k4)
        self.mlp = eqx.nn.MLP(
            hidden_dim,
            mlp_output_dim,
            width_size=hidden_dim,
            depth=2,
            key=k5,
        )

    def __call__(self, x: Array, theta: Array, context: Array, missing: Array) -> Array:
        p_x = self.proj_x(x)
        p_theta = self.proj_theta(theta)
        p_context = self.proj_context(context)
        p_missing = self.proj_missing(missing)
        combined = p_x + p_theta + p_context + p_missing
        return self.mlp(combined)

class VectorToVector(Protocol):
    def __call__(self, x: Array) -> Array: ...


class AutoregressiveApproximation(AmortizedVariationalApproximation):
    """Base class for autoregressive variational samplers."""

    target_struct_cls: Type[seqjtyping.Packable]
    context_dim: int
    condition_dim: int
    parameter_dim: int
    lag_order: int

    def __init__(
        self,
        target_struct_cls,
        *,
        sample_length: int,
        latent_context_dims: LatentContextDims,
        lag_order: int,
    ) -> None:
        super().__init__(
            target_struct_cls,
            (sample_length, target_struct_cls.flat_dim),
            sample_length,
        )
        self.context_dim = latent_context_dims.sequence_embedded_context_dim
        self.parameter_dim = latent_context_dims.parameter_context_dim
        self.condition_dim = latent_context_dims.condition_context_dim
        self.lag_order = lag_order

    def conditional(
        self,
        key: PRNGKeyArray,
        prev_x: tuple[Float[Array, " x_dim"], ...],
        previous_available_flag: Bool[Array, " lag_order"],
        theta_context: Float[Array, " param_dim"],
        context: Float[Array, " context_dim"],
        condition_context: Float[Array, " condition_dim"],
    ) -> tuple[Float[Array, " x_dim"], Float[Array, ""]]:
        raise NotImplementedError
        
    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: LatentContext,
        state: typing.Any = None,
    ) -> tuple[seqjtyping.Packable, Float[Array, " sample_length"], typing.Any]:
        parameter_context = condition.parameter_context.ravel().flatten()
        initial_prev_x_key, sample_key = jrandom.split(key, 2)

        def update(carry, key_context):
            key, ctx, condition = key_context
            ix, prev_x = carry
            previous_available_flag = (
                jnp.arange(self.lag_order) + ix - self.lag_order >= 0
            )
            next_x, log_q_x_ix = self.conditional(
                key, 
                prev_x, 
                previous_available_flag, 
                parameter_context, 
                ctx.ravel().flatten(), 
                condition.ravel().flatten()
            )
            next_x_context = (*prev_x[1:], next_x)
            return (ix + 1, next_x_context), (next_x, log_q_x_ix)

        init_state = self.initial_x_context(initial_prev_x_key, condition)

        _, (x_path, log_q_x_path) = jax.lax.scan(
            update,
            (0, init_state),
            (
                jrandom.split(sample_key, self.shape[0]), 
                condition.sequence_embedded_context,
                condition.condition_context
            ),
        )
        sample = self.target_struct_cls.unravel(x_path)
        log_q = jnp.sum(jnp.sum(log_q_x_path, axis=-1))
        return sample, log_q, state

    def initial_x_context(
        self,
        key: PRNGKeyArray,
        condition: LatentContext,
    ) -> tuple[Float[Array, " x_dim"], ...]:
        """Return initial context of zeros."""
        return tuple(
            jnp.zeros((self.shape[1],), dtype=jnp.float32)
            for _ in range(self.lag_order)
        )


class RandomAutoregressor(AutoregressiveApproximation):
    """Autoregressor that samples from standard normal regardless of context."""

    def conditional(self, key, prev_x, previous_available_flag, theta_context, context):
        return jrandom.normal(key, (self.shape[1],)), jrandom.normal(key, ())


class AmortizedUnivariateAutoregressor(AutoregressiveApproximation):
    amortizer_mlp: VectorToVector

    def __init__(
        self,
        target_struct_cls,
        *,
        sample_length: int,
        latent_context_dims: LatentContextDims,
        lag_order: int,
        nn_width: int,
        nn_depth: int,
        key: PRNGKeyArray,
    ) -> None:
        if target_struct_cls.flat_dim != 1:
            raise ValueError(
                "AmortizedUnivariateAutoregressor requires target_struct_cls.flat_dim == 1."
            )
        super().__init__(
            target_struct_cls,
            sample_length=sample_length,
            latent_context_dims=latent_context_dims,
            lag_order=lag_order,
        )
        input_dim = lag_order * 2 + self.context_dim + self.parameter_dim + self.condition_dim
        self.amortizer_mlp = ResNetMLP(
            in_size=input_dim,
            width=nn_width,
            out_size=2,
            depth=nn_depth,
            use_batchnorm=False,
            key=key,
        )
        # self.amortizer_mlp = eqx.nn.MLP(
        #     in_size=input_dim,
        #     width_size=nn_width,
        #     out_size=2,
        #     depth=nn_depth,
        #     # use_batchnorm=False,
        #     key=key,
        # )

    def conditional(
        self,
        key,
        prev_x,
        previous_available_flag,
        theta_context,
        context,
        condition_context,
    ):
        inputs = jnp.concatenate(
            [
                *prev_x,
                previous_available_flag,
                theta_context,
                context,
                condition_context,
            ]
        )
        z = jrandom.normal(key, shape=(1,))
        loc, _unc_scale = self.amortizer_mlp(inputs)
        scale = jax.nn.softplus(_unc_scale)
        x = z * scale + loc
        log_q_x = jstats.norm.logpdf(x, loc, scale)
        return x, log_q_x


class AmortizedInnovationUnivariateAutoregressor(AutoregressiveApproximation):
    amortizer_mlp: VectorToVector
    model: BayesianSequentialModelProtocol

    def __init__(
        self,
        model: BayesianSequentialModelProtocol,
        *,
        sample_length: int,
        latent_context_dims: LatentContextDims,
        lag_order: int,
        nn_width: int,
        nn_depth: int,
        key: PRNGKeyArray,
    ) -> None:
        if model.target.latent_cls.flat_dim != 1:
            raise ValueError(
                "AmortizedInnovationUnivariateAutoregressor requires latent flat_dim == 1."
            )
        super().__init__(
            model.target.latent_cls,
            sample_length=sample_length,
            latent_context_dims=latent_context_dims,
            lag_order=lag_order,
        )
        input_dim = lag_order * 2 + self.context_dim + self.parameter_dim + self.condition_dim
        self.amortizer_mlp = ResNetMLP(
            in_size=input_dim,
            width=nn_width,
            out_size=2,
            depth=nn_depth,
            use_batchnorm=False,
            key=key,
        )
        self.model = model

    def conditional(
        self,
        key,
        prev_x,
        previous_available_flag,
        theta_context,
        context,
        condition_context,
    ):
        inputs = jnp.concatenate(
            [
                *prev_x,
                previous_available_flag,
                theta_context,
                context,
                condition_context,
            ]
        )

        z = jrandom.normal(key, shape=(1,))
        loc, _unc_scale = self.amortizer_mlp(inputs)
        scale = jax.nn.softplus(_unc_scale)
        eps = z * scale + loc

        # this is a fixed reparameterization into the model prior
        latent_history = tuple(
            self.model.target.latent_t.unravel(px) for px in prev_x
        )
        model_params = self.model.parameterization.to_model_parameters(
            self.model.parameterization.inference_parameter_cls.unravel(theta_context)
        )
        condition = self.model.target.condition_cls.unravel(condition_context)
        prior_l, prior_scale = self.model.target.transition.loc_scale(
            latent_history,
            condition,
            jax.lax.stop_gradient(model_params),
        )

        x = prior_l + prior_scale * eps

        log_q_x = jstats.norm.logpdf(eps, loc, scale) - jnp.log(prior_scale)

        return x, log_q_x

    def initial_x_context(
        self,
        key,
        parameter_context,
        condition_context,
    ):
        model_params = self.model.convert_to_model_parameters(
            self.model.inference_parameter_cls.unravel(parameter_context[0])
        )

        condition = self.model.target.condition_cls.unravel(condition_context[0])

        #
        start = tuple(
            self.model.target.latent_cls.ravel(
                self.model.target.prior.sample(k, condition, model_params)[0]
            )
            for k in jrandom.split(key, self.lag_order)
        )

        return start


class AmortizedMultivariateAutoregressor(AutoregressiveApproximation):
    amortizer_mlp: VectorToVector
    _x_dim: int
    _n_tril: int

    def __init__(
        self,
        target_struct_cls,
        *,
        sample_length: int,
        latent_context_dims: LatentContextDims,
        lag_order: int,
        nn_width: int,
        nn_depth: int,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            target_struct_cls,
            sample_length=sample_length,
            latent_context_dims=latent_context_dims,
            lag_order=lag_order,
        )

        if self.shape[1] < 2:
            raise ValueError(
                "AmortizedMultivariateAutoregressor requires latent flat_dim >= 2"
            )

        self._x_dim = self.shape[1]
        self._n_tril = self._x_dim * (self._x_dim + 1) // 2
        input_dim = (
            lag_order * self._x_dim
            + lag_order
            + self.context_dim
            + self.parameter_dim
            + self.condition_dim
        )
        output_dim = self._x_dim + self._n_tril

        self.amortizer_mlp = ResNetMLP(
            in_size=input_dim,
            width=nn_width,
            out_size=output_dim,
            depth=nn_depth,
            use_batchnorm=False,
            key=key,
        )

    def _build_cholesky(self, unconstrained_chol: Array) -> Array:
        if unconstrained_chol.shape[-1] != self._n_tril:
            raise ValueError(
                "Invalid Cholesky parameter count: "
                f"expected {self._n_tril}, got {unconstrained_chol.shape[-1]}"
            )

        tri = jnp.zeros((self._x_dim, self._x_dim), dtype=unconstrained_chol.dtype)
        tril_ix = jnp.tril_indices(self._x_dim)
        tri = tri.at[tril_ix].set(unconstrained_chol)
        diag_ix = jnp.diag_indices(self._x_dim)
        tri = tri.at[diag_ix].set(jax.nn.softplus(tri[diag_ix]) + 1e-4)
        return tri

    def conditional(
        self,
        key,
        prev_x,
        previous_available_flag,
        theta_context,
        context,
        condition_context,
    ):
        flat_prev_x = [jnp.ravel(x) for x in prev_x]
        inputs = jnp.concatenate(
            [
                *flat_prev_x,
                previous_available_flag.astype(jnp.float32),
                theta_context,
                context,
                condition_context,
            ]
        )

        z = jrandom.normal(key, shape=(self._x_dim,))
        trans_params = self.amortizer_mlp(inputs)
        loc = trans_params[: self._x_dim]
        unconstrained_chol = trans_params[self._x_dim :]

        chol = self._build_cholesky(unconstrained_chol)

        x = chol @ z + loc
        log_det_chol = jnp.sum(jnp.log(jnp.diagonal(chol)))
        quadratic = jnp.sum(z**2)
        log_q_x = -0.5 * (self._x_dim * _LOG_2PI + 2.0 * log_det_chol + quadratic)
        return x, log_q_x
