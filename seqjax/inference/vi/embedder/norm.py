import typing
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


NormalizationKind = typing.Literal["none", "ema"]
ObservationNormalizationKind = typing.Literal["none", "conditional-ema"]


@dataclass(frozen=True)
class NormalizationConfig:
    """Normalization applied before an embedder constructs inference features.

    ``none`` is deliberately the default for every input so existing experiment
    configurations retain their previous feature shapes and values.
    """

    observation: ObservationNormalizationKind = "none"
    condition: NormalizationKind = "none"
    parameter: NormalizationKind = "none"
    momentum: float = 0.99
    eps: float = 1e-5


class FeatureNormalizer(typing.Protocol):
    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        *,
        reduce_axes: tuple[str, ...],
        training: bool,
    ) -> tuple[Array, eqx.nn.State]: ...


class ConditionalObservationNormalizer(typing.Protocol):
    def __call__(
        self,
        observations: Array,
        conditions: Array,
        parameters: Array,
        state: eqx.nn.State,
        *,
        reduce_axes: tuple[str, ...],
        training: bool,
    ) -> tuple[Array, eqx.nn.State]: ...


class EMAFeatureNorm(eqx.Module):
    """EMA standardization for arrays whose final axis is the feature axis."""

    index: eqx.nn.StateIndex
    momentum: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, feature_dim: int, momentum: float = 0.99, eps: float = 1e-5):
        if feature_dim < 0:
            raise ValueError(f"feature_dim must be >= 0, got {feature_dim}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.index = eqx.nn.StateIndex(
            {
                "mean": jnp.zeros((feature_dim,)),
                "var": jnp.ones((feature_dim,)),
                "initialized": jnp.array(False),
            }
        )
        self.momentum = momentum
        self.eps = eps

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        *,
        reduce_axes: tuple[str, ...],
        training: bool,
    ) -> tuple[Array, eqx.nn.State]:
        stats = state.get(self.index)
        mean = stats["mean"]
        var = stats["var"]
        initialized = stats["initialized"]

        local_axes = tuple(range(x.ndim - 1))
        batch_mean = jnp.mean(x, axis=local_axes) if local_axes else x
        batch_mean = jax.lax.pmean(batch_mean, axis_name=reduce_axes)
        local_var = (
            jnp.mean((x - batch_mean) ** 2, axis=local_axes)
            if local_axes
            else (x - batch_mean) ** 2
        )
        batch_var = jax.lax.pmean(local_var, axis_name=reduce_axes)

        active_mean = jnp.where(initialized, mean, batch_mean)
        active_var = jnp.where(initialized, var, batch_var)
        normalized = (x - active_mean) / jnp.sqrt(active_var + self.eps)

        new_stats = {
            "mean": jnp.where(
                initialized,
                self.momentum * mean + (1.0 - self.momentum) * batch_mean,
                batch_mean,
            ),
            "var": jnp.where(
                initialized,
                self.momentum * var + (1.0 - self.momentum) * batch_var,
                batch_var,
            ),
            "initialized": jnp.array(True),
        }
        if training:
            state = state.set(self.index, new_stats)
        return normalized, state


class ConditionalEMAObservationNorm(eqx.Module):
    """EMA observation normalization modulated by condition and parameters.

    The conditional affine map is initialized as the identity. This makes the
    initial behavior ordinary EMA standardization while allowing training to
    learn conditional location and log-scale corrections.
    """

    ema: EMAFeatureNorm
    affine: eqx.nn.Linear
    observation_dim: int = eqx.field(static=True)

    def __init__(
        self,
        observation_dim: int,
        condition_dim: int,
        parameter_dim: int,
        *,
        momentum: float,
        eps: float,
        key: jax.Array,
    ):
        self.ema = EMAFeatureNorm(observation_dim, momentum=momentum, eps=eps)
        affine = eqx.nn.Linear(
            condition_dim + parameter_dim,
            2 * observation_dim,
            key=key,
        )
        self.affine = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            affine,
            (jnp.zeros_like(affine.weight), jnp.zeros_like(affine.bias)),
        )
        self.observation_dim = observation_dim

    def __call__(
        self,
        observations: Array,
        conditions: Array,
        parameters: Array,
        state: eqx.nn.State,
        *,
        reduce_axes: tuple[str, ...],
        training: bool,
    ) -> tuple[Array, eqx.nn.State]:
        if observations.ndim != 2:
            raise ValueError(
                "conditional observation normalization requires observations "
                f"with shape (time, features), got {observations.shape}"
            )
        if conditions.shape[0] != observations.shape[0]:
            raise ValueError(
                "conditions and observations must have the same time dimension, "
                f"got {conditions.shape[0]} and {observations.shape[0]}"
            )
        standardized, state = self.ema(
            observations,
            state,
            reduce_axes=reduce_axes,
            training=training,
        )
        broadcast_parameters = jnp.broadcast_to(
            parameters,
            (observations.shape[0], parameters.shape[-1]),
        )
        conditioning = jnp.concatenate([conditions, broadcast_parameters], axis=-1)
        affine = jax.vmap(self.affine)(conditioning)
        shift, log_scale = jnp.split(affine, 2, axis=-1)
        log_scale = jnp.clip(log_scale, -10.0, 10.0)
        return (standardized - shift) * jnp.exp(-log_scale), state


class InputNormalization(eqx.Module):
    observation: None | ConditionalObservationNormalizer
    condition: None | FeatureNormalizer
    parameter: None | FeatureNormalizer
    include_condition: bool = eqx.field(static=True)
    include_parameter: bool = eqx.field(static=True)

    def __init__(
        self,
        config: NormalizationConfig,
        *,
        observation_dim: int,
        condition_dim: int,
        parameter_dim: int,
        key: jax.Array,
    ):
        if config.observation == "conditional-ema":
            self.observation = ConditionalEMAObservationNorm(
                observation_dim,
                condition_dim,
                parameter_dim,
                momentum=config.momentum,
                eps=config.eps,
                key=key,
            )
        elif config.observation == "none":
            self.observation = None
        else:
            raise ValueError(
                f"unknown observation normalization: {config.observation!r}"
            )

        if config.condition == "ema":
            self.condition = EMAFeatureNorm(
                condition_dim, momentum=config.momentum, eps=config.eps
            )
        elif config.condition == "none":
            self.condition = None
        else:
            raise ValueError(f"unknown condition normalization: {config.condition!r}")

        if config.parameter == "ema":
            self.parameter = EMAFeatureNorm(
                parameter_dim, momentum=config.momentum, eps=config.eps
            )
        elif config.parameter == "none":
            self.parameter = None
        else:
            raise ValueError(f"unknown parameter normalization: {config.parameter!r}")

        self.include_condition = self.condition is not None
        self.include_parameter = self.parameter is not None

    def normalize(
        self,
        observations: Array,
        conditions: Array,
        parameters: Array,
        state: eqx.nn.State | None,
        *,
        reduce_axes: tuple[str, ...],
        training: bool,
    ) -> tuple[Array, Array, Array, eqx.nn.State | None]:
        if (
            self.observation is not None
            or self.condition is not None
            or self.parameter is not None
        ) and state is None:
            raise ValueError(
                "normalization state is required when a non-null input normalizer is configured"
            )
        if state is None:
            return observations, conditions, parameters, state

        normalized_conditions = conditions
        normalized_parameters = parameters
        if self.condition is not None:
            normalized_conditions, state = self.condition(
                conditions, state, reduce_axes=reduce_axes, training=training
            )
        if self.parameter is not None:
            normalized_parameters, state = self.parameter(
                parameters, state, reduce_axes=reduce_axes, training=training
            )
        if self.observation is not None:
            observations, state = self.observation(
                observations,
                normalized_conditions,
                normalized_parameters,
                state,
                reduce_axes=reduce_axes,
                training=training,
            )
        return observations, normalized_conditions, normalized_parameters, state
