from typing import Tuple, Type

from seqjax.inference.vi.base import AmortizedVariationalApproximation
from seqjax.model.base import BayesianSequentialModel
import seqjax.model.typing
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, Bool, Float, PRNGKeyArray


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


def flat_to_chol(flat: Array, dim: int) -> Tuple[Array, Array]:
    tri = jnp.zeros((dim, dim))
    idx = jnp.tril_indices(dim)
    tri = tri.at[idx].set(flat)
    cov = tri @ tri.T
    return tri, cov


class AutoregressiveApproximation(AmortizedVariationalApproximation):
    """Base class for autoregressive variational samplers."""

    target_struct_cls: Type[seqjax.model.typing.Packable]
    buffer_length: int
    batch_length: int
    context_dim: int
    condition_dim: int
    parameter_dim: int
    lag_order: int

    def __init__(
        self,
        target_struct_cls,
        batch_length: int,
        buffer_length: int,
        context_dim: int,
        parameter_dim: int,
        condition_dim: int,
        lag_order: int,
    ) -> None:
        sample_length = 2 * buffer_length + batch_length
        super().__init__(
            target_struct_cls,
            (sample_length, target_struct_cls.flat_dim),
            batch_length,
            buffer_length,
        )
        self.context_dim = context_dim
        self.parameter_dim = parameter_dim
        self.condition_dim = condition_dim
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

    def sample_sub_path(
        self,
        key: PRNGKeyArray,
        theta_context: Float[Array, "sample_length param_dim"],
        context: Float[Array, "sample_length context_dim"],
        condition_context: Float[Array, "sample_length condition_dim"],
        num_steps: int,
        offset: int,
        init: tuple[Array, ...],
    ) -> tuple[seqjax.model.typing.Packable, Float[Array, " sample_length"]]:
        def update(carry, key_context):
            key, ctx, step_theta_context, condition = key_context
            ix, prev_x = carry
            previous_available_flag = (
                jnp.arange(self.lag_order) + ix - self.lag_order >= 0
            )
            next_x, log_q_x_ix = self.conditional(
                key, prev_x, previous_available_flag, step_theta_context, ctx, condition
            )
            next_x_context = (*prev_x[1:], next_x)
            return (ix + 1, next_x_context), (next_x, log_q_x_ix)

        init_state = (offset, init)

        keys = jrandom.split(key, num_steps)
        subpath_context = context[offset : offset + num_steps]
        subpath_theta_context = theta_context[offset : offset + num_steps]
        subpath_condition_context = condition_context[offset : offset + num_steps]
        target_condition_shape = (
            subpath_context.shape[:-1] + subpath_condition_context.shape[-1:]
        )  # keep condition last dim, take context leading dims
        subpath_condition_context = jnp.broadcast_to(
            subpath_condition_context, target_condition_shape
        )

        _, (x_path, log_q_x_path) = jax.lax.scan(
            update,
            init_state,
            (keys, subpath_context, subpath_theta_context, subpath_condition_context),
        )
        return self.target_struct_cls.unravel(x_path), jnp.sum(log_q_x_path, axis=-1)

    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: tuple[
            Float[Array, " sample_length param_dim"],
            Float[Array, " sample_length context_dim"],
            Float[Array, " sample_length condition_dim"],
        ],
    ) -> tuple[seqjax.model.typing.Packable, Float[Array, " sample_length"]]:
        parameter_context, observation_context, condition_context = condition
        initial_prev_x_key, sample_key = jrandom.split(key, 2)
        x_path, log_q_x_path = self.sample_sub_path(
            sample_key,
            parameter_context,
            observation_context,
            condition_context,
            self.shape[0],
            0,
            self.initial_x_context(
                initial_prev_x_key, parameter_context, condition_context
            ),
        )
        return x_path, jnp.sum(log_q_x_path)

    def initial_x_context(
        self,
        key: PRNGKeyArray,
        parameter_context: PRNGKeyArray,
        condition_context: PRNGKeyArray,
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
    amortizer_mlp: eqx.nn.MLP | ResNetMLP

    def __init__(
        self,
        target_struct_cls,
        *,
        buffer_length: int,
        batch_length: int,
        context_dim: int,
        parameter_dim: int,
        condition_dim: int,
        lag_order: int,
        nn_width: int,
        nn_depth: int,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            target_struct_cls,
            buffer_length=buffer_length,
            batch_length=batch_length,
            context_dim=context_dim,
            parameter_dim=parameter_dim,
            condition_dim=condition_dim,
            lag_order=lag_order,
        )
        input_dim = lag_order * 2 + context_dim + parameter_dim + condition_dim
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
    amortizer_mlp: eqx.nn.MLP | ResNetMLP
    model: BayesianSequentialModel

    def __init__(
        self,
        model: BayesianSequentialModel,
        *,
        buffer_length: int,
        batch_length: int,
        context_dim: int,
        parameter_dim: int,
        condition_dim: int,
        lag_order: int,
        nn_width: int,
        nn_depth: int,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            model.target.latent_cls,
            buffer_length=buffer_length,
            batch_length=batch_length,
            context_dim=context_dim,
            parameter_dim=parameter_dim,
            condition_dim=condition_dim,
            lag_order=lag_order,
        )
        input_dim = lag_order * 2 + context_dim + parameter_dim + condition_dim
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
            self.model.target.transition.latent_t.unravel(px) for px in prev_x
        )
        model_params = self.model.convert_to_model_parameters(
            self.model.inference_parameter_cls.unravel(theta_context)
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


"""
Multivariate versions below are sketches,  not currently implemented
"""
# class AmortizedMultivariateAutoregressor(AutoregressiveApproximation):
#     amortizer_mlp: eqx.nn.MLP

#     def __init__(
#         self,
#         *,
#         sample_length: int,
#         context_dim: int,
#         parameter_dim: int,
#         lag_order: int,
#         nn_width: int,
#         nn_depth: int,
#         x_dim: int,
#         key: PRNGKeyArray,
#     ) -> None:
#         super().__init__(
#             sample_length=sample_length,
#             x_dim=x_dim,
#             context_dim=context_dim,
#             parameter_dim=parameter_dim,
#             lag_order=lag_order,
#         )
#         input_dim = lag_order * (1 + x_dim) + context_dim + parameter_dim
#         output_dim = x_dim + int(0.5 * x_dim * (x_dim + 1))
#         self.amortizer_mlp = eqx.nn.MLP(
#             in_size=input_dim,
#             out_size=output_dim,
#             width_size=nn_width,
#             depth=nn_depth,
#             key=key,
#         )

#     def conditional(self, key, prev_x, previous_available_flag, theta_context, context):  # noqa: D401, ANN001
#         flat_prev_x = (jnp.ravel(_x) for _x in prev_x)
#         inputs = jnp.concatenate(
#             [*flat_prev_x, previous_available_flag, theta_context, context]
#         )
#         z = jrandom.normal(key, shape=(self.x_dim,))
#         trans_params = self.amortizer_mlp(inputs)
#         loc = trans_params[: self.x_dim]
#         cholesky, cov = flat_to_chol(trans_params[self.x_dim :], self.x_dim)
#         x = cholesky @ z + loc
#         log_q_x = jstats.multivariate_normal.logpdf(x, loc, cov)
#         return x, log_q_x


# class AmortizedMultivariateIsotropicAutoregressor(AutoregressiveApproximation):
#     amortizer_mlp: eqx.nn.MLP

#     def __init__(
#         self,
#         *,
#         sample_length: int,
#         context_dim: int,
#         parameter_dim: int,
#         lag_order: int,
#         nn_width: int,
#         nn_depth: int,
#         x_dim: int,
#         key: PRNGKeyArray,
#     ) -> None:
#         super().__init__(
#             sample_length=sample_length,
#             x_dim=x_dim,
#             context_dim=context_dim,
#             parameter_dim=parameter_dim,
#             lag_order=lag_order,
#         )
#         input_dim = lag_order * (1 + x_dim) + context_dim + parameter_dim
#         output_dim = 2 * x_dim
#         self.amortizer_mlp = eqx.nn.MLP(
#             in_size=input_dim,
#             out_size=output_dim,
#             width_size=nn_width,
#             depth=nn_depth,
#             key=key,
#         )

#     def conditional(self, key, prev_x, previous_available_flag, theta_context, context):  # noqa: D401, ANN001
#         inputs = jnp.concatenate(
#             [*prev_x, previous_available_flag, theta_context, context]
#         )
#         z = jrandom.normal(key, shape=(self.x_dim,))
#         loc, _unc_scale = jnp.split(self.amortizer_mlp(inputs), [self.x_dim])
#         scale = jax.nn.softplus(_unc_scale)
#         x = z * scale + loc
#         log_q_x = jstats.norm.logpdf(x, loc, scale).sum()
#         return x, log_q_x
