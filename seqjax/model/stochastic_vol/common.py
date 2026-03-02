"""Common building blocks for stochastic-volatility model variants."""

from collections import OrderedDict
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    Emission,
    ParameterPrior,
    Prior,
    Transition,
)
from seqjax.model.typing import (
    Condition,
    NoCondition,
    HyperParameters,
    Observation,
    Parameters,
    Latent,
)


class LatentVol(Latent):
    """Latent state containing the log volatility."""

    log_vol: Scalar
    _shape_template = OrderedDict(
        log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LatentVar(Latent):
    """Latent state containing the log variance."""

    log_var: Scalar
    _shape_template = OrderedDict(
        log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogReturnObs(Observation):
    """Observed log return."""

    log_return: Scalar
    _shape_template: ClassVar = OrderedDict(
        log_return=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVolRW(Parameters):
    """Parameters for a log-volatility random walk."""

    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        mean_reversion=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarParams(Parameters):
    """Parameters for a log-variance random walk with AR dynamics."""

    std_log_var: Scalar
    ar: Scalar
    long_term_log_var: Scalar
    skew: Scalar
    _shape_template = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        skew=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarStd(Parameters):
    std_log_var: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarAR(Parameters):
    ar: Scalar
    _shape_template: ClassVar = OrderedDict(
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LVolStd(Parameters):
    std_log_vol: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


def lvar_from_std_only(lv_only: LogVarStd, ref_params: LogVarParams) -> LogVarParams:
    return LogVarParams(
        std_log_var=lv_only.std_log_var,
        ar=jnp.ones_like(lv_only.std_log_var) * ref_params.ar,
        long_term_log_var=jnp.ones_like(lv_only.std_log_var)
        * ref_params.long_term_log_var,
        skew=jnp.ones_like(lv_only.std_log_var) * ref_params.skew,
    )


def lvar_from_ar_only(ar_only: LogVarAR, ref_params: LogVarParams) -> LogVarParams:
    return LogVarParams(
        std_log_var=jnp.ones_like(ar_only.ar) * ref_params.std_log_var,
        ar=ar_only.ar,
        long_term_log_var=jnp.ones_like(ar_only.ar) * ref_params.long_term_log_var,
        skew=jnp.ones_like(ar_only.ar) * ref_params.skew,
    )


def lv_to_std_only(lv_only: LVolStd, ref_params: LogVolRW) -> LogVolRW:
    return LogVolRW(
        std_log_vol=lv_only.std_log_vol,
        long_term_vol=jnp.ones_like(lv_only.std_log_vol) * ref_params.long_term_vol,
        mean_reversion=jnp.ones_like(lv_only.std_log_vol) * ref_params.mean_reversion,
    )


class TimeIncrement(Condition):
    dt: Scalar
    _shape_template: ClassVar = OrderedDict(
        dt=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class StochVarPrior(ParameterPrior[LogVarStd, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LogVarStd:
        _ = hyperparameters
        return LogVarStd(std_log_var=10 / jrandom.gamma(key, 10))

    @staticmethod
    def log_prob(parameters: LogVarStd, hyperparameters: HyperParameters) -> Scalar:
        _ = hyperparameters
        return (
            jstats.gamma.logpdf(1 / parameters.std_log_var, 10, scale=1 / 10)
            - 2 * jnp.log(parameters.std_log_var)
        )


class StochVarARPrior(ParameterPrior[LogVarAR, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LogVarAR:
        _ = hyperparameters
        return LogVarAR(ar=2 * jax.random.beta(key, 20, 1.5) - 1)

    @staticmethod
    def log_prob(parameters: LogVarAR, hyperparameters: HyperParameters) -> Scalar:
        _ = hyperparameters
        return jstats.beta.logpdf((parameters.ar + 1) / 2, 20, 1.5) - jnp.log(2)


class StochVarFullPrior(ParameterPrior[LogVarParams, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LogVarParams:
        _ = hyperparameters
        key_std, key_ar = jrandom.split(key)
        return LogVarParams(
            std_log_var=10 / jrandom.gamma(key_std, 10),
            ar=2 * jax.random.beta(key_ar, 20, 1.5) - 1,
            long_term_log_var=jnp.array(2 * jnp.log(0.2)),
            skew=jnp.array(0.0),
        )

    @staticmethod
    def log_prob(parameters: LogVarParams, hyperparameters: HyperParameters) -> Scalar:
        _ = hyperparameters
        return (
            jstats.gamma.logpdf(1 / parameters.std_log_var, 10, scale=1 / 10)
            - 2 * jnp.log(parameters.std_log_var)
            + jstats.beta.logpdf((parameters.ar + 1) / 2, 20, 1.5)
            - jnp.log(2)
        )


class StochVolParamPrior(ParameterPrior[LogVolRW, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LogVolRW:
        _ = hyperparameters
        k1, k2, k3 = jrandom.split(key, 3)
        return LogVolRW(
            std_log_vol=10 / jrandom.gamma(k1, 10),
            mean_reversion=0.5 * jrandom.gamma(k2, 2.0),
            long_term_vol=10 / jrandom.gamma(k3, 10),
        )

    @staticmethod
    def log_prob(parameters: LogVolRW, hyperparameters: HyperParameters) -> Scalar:
        _ = hyperparameters
        return (
            jstats.gamma.logpdf(1 / parameters.std_log_vol, 10, scale=1 / 10)
            - 2 * jnp.log(parameters.std_log_vol)
            + jstats.gamma.logpdf(parameters.mean_reversion, 2.0, scale=0.5)
            + jstats.gamma.logpdf(1 / parameters.long_term_vol, 10, scale=1 / 10)
            - 2 * jnp.log(parameters.long_term_vol)
        )


class StdLogVolPrior(ParameterPrior[LVolStd, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LVolStd:
        _ = hyperparameters
        return LVolStd(std_log_vol=10 / jrandom.gamma(key, 10))

    @staticmethod
    def log_prob(parameters: LVolStd, hyperparameters: HyperParameters) -> Scalar:
        _ = hyperparameters
        return (
            jstats.gamma.logpdf(1 / parameters.std_log_vol, 10, scale=1 / 10)
            - 2 * jnp.log(parameters.std_log_vol)
        )


def gaussian_var_two_step_sample(
    key: PRNGKeyArray,
    conditions: tuple[NoCondition, NoCondition],
    parameters: LogVarParams,
) -> tuple[LatentVar, LatentVar]:
    _ = conditions
    key_1, key_2 = jrandom.split(key)
    first = LatentVar(log_var=parameters.long_term_log_var)
    second = LatentVar(
        log_var=parameters.long_term_log_var
        + parameters.std_log_var * jrandom.normal(key_2)
    )
    _ = key_1
    return first, second


def gaussian_var_two_step_log_prob(
    latent: tuple[LatentVar, LatentVar],
    conditions: tuple[NoCondition, NoCondition],
    parameters: LogVarParams,
) -> Scalar:
    _ = conditions
    first, second = latent
    first_log = jnp.where(
        jnp.isclose(first.log_var, parameters.long_term_log_var),
        0.0,
        -jnp.inf,
    )
    second_log = jstats.norm.logpdf(
        second.log_var,
        loc=parameters.long_term_log_var,
        scale=parameters.std_log_var,
    )
    return first_log + second_log


gaussian_var_two_step_start = Prior[
    tuple[LatentVar, LatentVar], tuple[NoCondition, NoCondition], LogVarParams
](
    order=2,
    sample=gaussian_var_two_step_sample,
    log_prob=gaussian_var_two_step_log_prob,
)


def _random_walk_ar_loc_scale(
    latent_history: tuple[LatentVar, LatentVar],
    condition: NoCondition,
    parameters: LogVarParams,
) -> tuple[Scalar, Scalar]:
    _ = condition
    _, current = latent_history
    loc = parameters.long_term_log_var + parameters.ar * (
        current.log_var - parameters.long_term_log_var
    )
    return loc, parameters.std_log_var


def random_walk_ar_sample(
    key: PRNGKeyArray,
    latent_history: tuple[LatentVar, LatentVar],
    condition: NoCondition,
    parameters: LogVarParams,
) -> LatentVar:
    loc, scale = _random_walk_ar_loc_scale(latent_history, condition, parameters)
    return LatentVar(log_var=loc + scale * jrandom.normal(key))


def random_walk_ar_log_prob(
    latent_history: tuple[LatentVar, LatentVar],
    latent: LatentVar,
    condition: NoCondition,
    parameters: LogVarParams,
) -> Scalar:
    loc, scale = _random_walk_ar_loc_scale(latent_history, condition, parameters)
    return jstats.norm.logpdf(latent.log_var, loc=loc, scale=scale)


random_walk_ar = Transition[
    tuple[LatentVar, LatentVar],
    LatentVar,
    NoCondition,
    LogVarParams,
](
    order=2,
    sample=random_walk_ar_sample,
    log_prob=random_walk_ar_log_prob,
)


def var_return_mean_and_scale(
    latent: tuple[LatentVar, LatentVar],
    condition: NoCondition,
    parameters: LogVarParams,
) -> tuple[Scalar, Scalar]:
    _ = condition
    previous_latent, current_latent = latent
    current_var = jnp.exp(current_latent.log_var / 2)
    previous_var = jnp.exp(previous_latent.log_var / 2)
    mean = (
        parameters.ar
        * (current_latent.log_var - parameters.long_term_log_var)
        * previous_var
    )
    scale = current_var * jnp.sqrt(1 - parameters.ar**2)
    return mean, scale


def log_return_var_sample(
    key: PRNGKeyArray,
    latent: tuple[LatentVar, LatentVar],
    observation_history: tuple[()],
    condition: NoCondition,
    parameters: LogVarParams,
) -> LogReturnObs:
    _ = observation_history
    mean, scale = var_return_mean_and_scale(latent, condition, parameters)
    return LogReturnObs(log_return=mean + scale * jrandom.normal(key))


def log_return_var_prob(
    latent: tuple[LatentVar, LatentVar],
    observation: LogReturnObs,
    observation_history: tuple[()],
    condition: NoCondition,
    parameters: LogVarParams,
) -> Scalar:
    _ = observation_history
    mean, scale = var_return_mean_and_scale(latent, condition, parameters)
    return jstats.norm.logpdf(observation.log_return, loc=mean, scale=scale)


log_return_var = Emission[
    tuple[LatentVar, LatentVar],
    NoCondition,
    LogReturnObs,
    LogVarParams,
](
    order=2,
    sample=log_return_var_sample,
    log_prob=log_return_var_prob,
)


def gaussian_start_sample(
    key: PRNGKeyArray,
    conditions: tuple[TimeIncrement],
    parameters: LogVolRW,
) -> tuple[LatentVol]:
    _ = conditions
    return (LatentVol(log_vol=jnp.log(parameters.long_term_vol)),)


def gaussian_start_log_prob(
    latent: tuple[LatentVol],
    conditions: tuple[TimeIncrement],
    parameters: LogVolRW,
) -> Scalar:
    _ = conditions
    prior_mean = jnp.log(parameters.long_term_vol)
    return jstats.norm.logpdf(latent[0].log_vol, loc=prior_mean, scale=0.1)


gaussian_start = Prior[tuple[LatentVol], tuple[TimeIncrement], LogVolRW](
    order=1,
    sample=gaussian_start_sample,
    log_prob=gaussian_start_log_prob,
)


def _random_walk_loc_scale(
    latent_history: tuple[LatentVol],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> tuple[Scalar, Scalar]:
    (current_latent,) = latent_history
    dt = condition.dt
    reversion = jnp.exp(-parameters.mean_reversion * dt)
    loc = (1 - reversion) * jnp.log(parameters.long_term_vol) + reversion * current_latent.log_vol
    scale = parameters.std_log_vol * jnp.sqrt(dt)
    return loc, scale


def random_walk_sample(
    key: PRNGKeyArray,
    latent_history: tuple[LatentVol],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> LatentVol:
    loc, scale = _random_walk_loc_scale(latent_history, condition, parameters)
    return LatentVol(log_vol=loc + scale * jrandom.normal(key))


def random_walk_log_prob(
    latent_history: tuple[LatentVol],
    latent: LatentVol,
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> Scalar:
    loc, scale = _random_walk_loc_scale(latent_history, condition, parameters)
    return jstats.norm.logpdf(latent.log_vol, loc=loc, scale=scale)


random_walk = Transition[
    tuple[LatentVol],
    LatentVol,
    TimeIncrement,
    LogVolRW,
](
    order=1,
    sample=random_walk_sample,
    log_prob=random_walk_log_prob,
)


def log_return_sample(
    key: PRNGKeyArray,
    latent: tuple[LatentVol],
    observation_history: tuple[()],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> LogReturnObs:
    _ = observation_history, condition
    (current_latent,) = latent
    return_scale = jnp.exp(current_latent.log_vol)
    return LogReturnObs(log_return=jrandom.normal(key) * return_scale)


def log_return_log_prob(
    latent: tuple[LatentVol],
    observation: LogReturnObs,
    observation_history: tuple[()],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> Scalar:
    _ = observation_history, condition, parameters
    (current_latent,) = latent
    return_scale = jnp.exp(current_latent.log_vol)
    return jstats.norm.logpdf(observation.log_return, loc=0.0, scale=return_scale)


log_return = Emission[tuple[LatentVol], TimeIncrement, LogReturnObs, LogVolRW](
    order=1,
    sample=log_return_sample,
    log_prob=log_return_log_prob,
)


def make_constant_time_increments(
    sequence_length: int,
    *,
    dt: float = 1.0,
    prior_order: int = 1,
) -> TimeIncrement:
    """Return a ``TimeIncrement`` tree filled with a constant ``dt``."""

    if sequence_length < 1:
        raise ValueError(
            f"sequence_length must be >= 1, got {sequence_length}",
        )

    required_length = sequence_length + prior_order - 1
    dt_value = jnp.asarray(dt, dtype=jnp.float32)
    increments = jnp.full((required_length,), dt_value, dtype=dt_value.dtype)
    return TimeIncrement(dt=increments)
