from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Scalar

from seqjax.model.typing import Condition, Latent, Observation, Parameters


class LatentVol(Latent):
    """Latent state containing log-volatility."""

    log_vol: Scalar
    _shape_template: ClassVar = OrderedDict(
        log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LatentVar(Latent):
    """Latent state containing log-variance."""

    log_var: Scalar
    _shape_template: ClassVar = OrderedDict(
        log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogReturnObs(Observation):
    """Observed log return."""

    log_return: Scalar
    _shape_template: ClassVar = OrderedDict(
        log_return=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVolRW(Parameters):
    """Parameters for mean-reverting random walk on log-volatility."""

    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        mean_reversion=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVolWithSkew(Parameters):
    """Random-walk parameters including skew (return/log-vol correlation)."""

    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar
    skew: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        mean_reversion=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        skew=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LVolStd(Parameters):
    """Inference parameters for std-only stochastic-volatility fitting."""

    std_log_vol: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarParams(Parameters):
    """Parameters for AR(1) dynamics on log-variance."""

    std_log_var: Scalar
    ar: Scalar
    long_term_log_var: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarStd(Parameters):
    """Inference parameters for std-only stochastic-variance fitting."""

    std_log_var: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarAR(Parameters):
    """Inference parameters for ar-only stochastic-variance fitting."""

    ar: Scalar
    _shape_template: ClassVar = OrderedDict(
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class TimeIncrement(Condition):
    """Time increment between observations."""

    dt: Scalar
    _shape_template: ClassVar = OrderedDict(
        dt=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )
