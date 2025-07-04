from __future__ import annotations

from typing import Tuple

from seqjax.inference.embedder import Embedder
from seqjax.model.base import (
    SequentialModel,
    ParticleType,
    ObservationType,
    ConditionType,
    ParametersType,
)
from seqjax.model.typing import Batched, SequenceAxis

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jax import flatten_util
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

    layer = eqx.tree_at(lambda l: l.weight, layer, new_weight)
    layer = eqx.tree_at(lambda l: l.bias, layer, new_bias)
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


class Residual(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, mlp: eqx.nn.MLP) -> None:
        self.mlp = mlp

    def __call__(self, x: Array, *args, **kwargs) -> Array:  # noqa: ANN001
        return self.mlp(x, *args, **kwargs) + x


def flat_to_chol(flat: Array, dim: int) -> Tuple[Array, Array]:
    tri = jnp.zeros((dim, dim))
    idx = jnp.tril_indices(dim)
    tri = tri.at[idx].set(flat)
    cov = tri @ tri.T
    return tri, cov


class AutoregressiveSampler(eqx.Module):
    """Minimal base class for autoregressive samplers."""

    sample_length: int
    x_dim: int
    context_dim: int
    parameter_dim: int

    def __init__(
        self, *, sample_length: int, x_dim: int, context_dim: int, parameter_dim: int
    ) -> None:
        self.sample_length = sample_length
        self.x_dim = x_dim
        self.context_dim = context_dim
        self.parameter_dim = parameter_dim


class Autoregressor(AutoregressiveSampler):
    """Base class for autoregressive variational samplers."""

    lag_order: int

    def __init__(
        self,
        *,
        sample_length: int,
        x_dim: int,
        context_dim: int,
        parameter_dim: int,
        lag_order: int = 1,
    ) -> None:
        super().__init__(
            sample_length=sample_length,
            x_dim=x_dim,
            context_dim=context_dim,
            parameter_dim=parameter_dim,
        )
        self.lag_order = lag_order
        assert lag_order > 0, "lag must be > 0"

    def conditional(
        self,
        key: PRNGKeyArray,
        prev_x: tuple[Float[Array, "x_dim"], ...],
        previous_available_flag: Bool[Array, "lag_order"],
        theta_context: Float[Array, "param_dim"],
        context: Float[Array, "context_dim"],
    ) -> tuple[Float[Array, "x_dim"], Float[Array, ""]]:
        raise NotImplementedError

    def sample_sub_path(
        self,
        key: PRNGKeyArray,
        theta_context: Float[Array, "param_dim"],
        context: Float[Array, "sample_length context_dim"],
        num_steps: int,
        offset: int,
        init: tuple[Array, ...],
    ) -> tuple[Float[Array, "sample_length x_dim"], Float[Array, "sample_length"]]:
        def update(carry, key_context):
            key, ctx = key_context
            ix, prev_x = carry
            previous_available_flag = (
                jnp.arange(self.lag_order) + ix - self.lag_order >= 0
            )
            next_x, log_q_x_ix = self.conditional(
                key, prev_x, previous_available_flag, theta_context, ctx
            )
            next_x_context = (*prev_x[1:], next_x)
            return (ix + 1, next_x_context), (next_x, log_q_x_ix)

        init_state = (offset, init)
        keys = jrandom.split(key, num_steps)
        subpath_context = context[offset : offset + num_steps]
        _, (x_path, log_q_x_path) = jax.lax.scan(
            update, init_state, (keys, subpath_context)
        )
        return x_path, jnp.sum(log_q_x_path, axis=-1)

    def sample_single_path(
        self,
        key: PRNGKeyArray,
        theta_context: Float[Array, "param_dim"],
        context: Float[Array, "sample_length context_dim"],
    ) -> tuple[Float[Array, "sample_length x_dim"], Float[Array, "sample_length"]]:
        return self.sample_sub_path(
            key,
            theta_context,
            context,
            self.sample_length,
            0,
            tuple(jnp.zeros(self.x_dim) for _ in range(self.lag_order)),
        )

    def sample_initial_state(
        self,
        key: PRNGKeyArray,
        theta_context: Float[Array, "param_dim"],
        context: Float[Array, "sample_length context_dim"],
    ) -> tuple[Float[Array, "sample_length x_dim"], Float[Array, "sample_length"]]:
        return self.sample_sub_path(
            key,
            theta_context,
            context,
            2,
            0,
            tuple(jnp.zeros(self.x_dim) for _ in range(self.lag_order)),
        )


class RandomAutoregressor(Autoregressor):
    """Autoregressor that samples from standard normal regardless of context."""

    def conditional(
        self, key, prev_x, previous_available_flag, theta_context, context
    ):  # noqa: D401, ANN001
        return jrandom.normal(key, (self.x_dim,)), jrandom.normal(key, ())


class AmortizedUnivariateAutoregressor(Autoregressor):
    amortizer_mlp: eqx.nn.MLP | ResNetMLP

    def __init__(
        self,
        *,
        sample_length: int,
        context_dim: int,
        parameter_dim: int,
        lag_order: int,
        nn_width: int,
        nn_depth: int,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            sample_length=sample_length,
            x_dim=1,
            context_dim=context_dim,
            parameter_dim=parameter_dim,
            lag_order=lag_order,
        )
        input_dim = lag_order * 2 + context_dim + parameter_dim
        self.amortizer_mlp = ResNetMLP(
            in_size=input_dim,
            width=nn_width,
            out_size=2,
            depth=nn_depth,
            use_batchnorm=False,
            key=key,
        )

    def conditional(
        self, key, prev_x, previous_available_flag, theta_context, context
    ):  # noqa: D401, ANN001
        inputs = jnp.concatenate(
            [*prev_x, previous_available_flag, theta_context, context]
        )
        z = jrandom.normal(key, shape=(1,))
        loc, _unc_scale = self.amortizer_mlp(inputs)
        scale = jnp.clip(jax.nn.softplus(_unc_scale), 1e-10, 1e2)
        x = z * scale + loc
        log_q_x = jstats.norm.logpdf(x, loc, scale)
        return x, log_q_x


class AmortizedMultivariateAutoregressor(Autoregressor):
    amortizer_mlp: eqx.nn.MLP

    def __init__(
        self,
        *,
        sample_length: int,
        context_dim: int,
        parameter_dim: int,
        lag_order: int,
        nn_width: int,
        nn_depth: int,
        x_dim: int,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            sample_length=sample_length,
            x_dim=x_dim,
            context_dim=context_dim,
            parameter_dim=parameter_dim,
            lag_order=lag_order,
        )
        input_dim = lag_order * (1 + x_dim) + context_dim + parameter_dim
        output_dim = x_dim + int(0.5 * x_dim * (x_dim + 1))
        self.amortizer_mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=output_dim,
            width_size=nn_width,
            depth=nn_depth,
            key=key,
        )

    def conditional(
        self, key, prev_x, previous_available_flag, theta_context, context
    ):  # noqa: D401, ANN001
        flat_prev_x = (jnp.ravel(_x) for _x in prev_x)
        inputs = jnp.concatenate(
            [*flat_prev_x, previous_available_flag, theta_context, context]
        )
        z = jrandom.normal(key, shape=(self.x_dim,))
        trans_params = self.amortizer_mlp(inputs)
        loc = trans_params[: self.x_dim]
        cholesky, cov = flat_to_chol(trans_params[self.x_dim :], self.x_dim)
        x = cholesky @ z + loc
        log_q_x = jstats.multivariate_normal.logpdf(x, loc, cov)
        return x, log_q_x


class AmortizedMultivariateIsotropicAutoregressor(Autoregressor):
    amortizer_mlp: eqx.nn.MLP

    def __init__(
        self,
        *,
        sample_length: int,
        context_dim: int,
        parameter_dim: int,
        lag_order: int,
        nn_width: int,
        nn_depth: int,
        x_dim: int,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            sample_length=sample_length,
            x_dim=x_dim,
            context_dim=context_dim,
            parameter_dim=parameter_dim,
            lag_order=lag_order,
        )
        input_dim = lag_order * (1 + x_dim) + context_dim + parameter_dim
        output_dim = 2 * x_dim
        self.amortizer_mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=output_dim,
            width_size=nn_width,
            depth=nn_depth,
            key=key,
        )

    def conditional(
        self, key, prev_x, previous_available_flag, theta_context, context
    ):  # noqa: D401, ANN001
        inputs = jnp.concatenate(
            [*prev_x, previous_available_flag, theta_context, context]
        )
        z = jrandom.normal(key, shape=(self.x_dim,))
        loc, _unc_scale = jnp.split(self.amortizer_mlp(inputs), [self.x_dim])
        scale = jax.nn.softplus(_unc_scale)
        x = z * scale + loc
        log_q_x = jstats.norm.logpdf(x, loc, scale).sum()
        return x, log_q_x


class AutoregressiveVIConfig(eqx.Module):
    """Configuration for :func:`run_autoregressive_vi`."""

    sampler: Autoregressor | None = None
    embedder: Embedder | None = None
    num_samples: int = 1
    return_parameters: bool = False
    parameter_std: float = 0.0


def run_autoregressive_vi(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    observations: Batched[ObservationType, SequenceAxis],
    *,
    parameters: ParametersType,
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    initial_latents: Batched[ParticleType, SequenceAxis] | None = None,
    config: AutoregressiveVIConfig,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_history: tuple[ObservationType, ...] | None = None,
) -> Batched[ParticleType, SequenceAxis | int]:
    """Sample latent paths using an autoregressive variational sampler."""

    sampler = config.sampler
    embedder = config.embedder
    if sampler is None:
        raise ValueError("sampler must be provided in config")
    if embedder is None:
        raise ValueError("embedder must be provided in config")

    obs_array = jnp.squeeze(observations.as_array(), -1)  # type: ignore[attr-defined]
    context = embedder.embed(obs_array)
    theta_flat, unravel = flatten_util.ravel_pytree(parameters)

    key_theta, key_samples = jrandom.split(key)
    sample_keys = jrandom.split(key_samples, config.num_samples)
    if config.parameter_std > 0:
        theta_noise = jrandom.normal(
            key_theta, shape=(config.num_samples, theta_flat.shape[0])
        )
        theta_samples_flat = theta_flat + config.parameter_std * theta_noise
    else:
        theta_samples_flat = jnp.broadcast_to(theta_flat, (config.num_samples, theta_flat.shape[0]))

    xs, _ = jax.vmap(sampler.sample_single_path, in_axes=[0, 0, None])(
        sample_keys, theta_samples_flat, context
    )
    latents = target.particle_type.from_array(xs)  # type: ignore[attr-defined]
    if config.return_parameters:
        param_samples = jax.vmap(unravel)(theta_samples_flat)
        return latents, param_samples
    return latents
