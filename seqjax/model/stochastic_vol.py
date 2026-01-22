"""Stochastic volatility models.
All values are in annualised terms.
"""

from collections import OrderedDict
from typing import ClassVar, Protocol
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    BayesianSequentialModel,
    BayesianSequentialModel_TO2_EO2,
    Emission,
    GaussianLocScaleTransition,
    ParameterPrior,
    Prior,
    SequentialModel,
    SequentialModel_TO2_EO2,
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


class LogVarAR(Parameters):
    """Parameters for a log-volatility random walk."""

    std_log_var: Scalar
    ar: Scalar
    _shape_template = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarStd(Parameters):
    """Parameters for a log-volatility random walk."""

    std_log_var: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


def lvar_to_std_only(lv_only: LogVarStd, ref_params: LogVarAR) -> LogVarAR:
    return LogVarAR(
        std_log_var=lv_only.std_log_var,
        ar=jnp.ones_like(lv_only.std_log_var) * ref_params.ar,
    )


class LVolStd(Parameters):
    """Parameters for a log-volatility random walk."""

    std_log_vol: Scalar
    # long_term_vol: Scalar
    _shape_template: ClassVar = OrderedDict(
        std_log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        # long_term_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVolWithSkew(Parameters):
    """Random-walk parameters including a skew term."""

    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar
    skew: Scalar  # correlation between random variations
    _shape_template: ClassVar = OrderedDict(
        std_log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        mean_reversion=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        skew=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


def lv_to_std_only(lv_only: LVolStd, ref_params: LogVolRW) -> LogVolRW:
    return LogVolRW(
        std_log_vol=lv_only.std_log_vol,
        # long_term_vol=lv_only.long_term_vol,
        long_term_vol=jnp.ones_like(lv_only.std_log_vol) * ref_params.long_term_vol,
        mean_reversion=jnp.ones_like(lv_only.std_log_vol) * ref_params.mean_reversion,
    )


class TimeIncrement(Condition):
    """Time step between observations."""

    dt: Scalar  # time since last observation
    _shape_template: ClassVar = OrderedDict(
        dt=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class StochVarPrior(ParameterPrior[LogVarStd, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LogVarStd:
        _ = hyperparameters  # unused
        # from aicher https://arxiv.org/pdf/1901.10568
        return LogVarStd(
            std_log_var=10 / jrandom.gamma(key, 10),
        )

    @staticmethod
    def log_prob(parameters: LogVarStd, hyperparameters: HyperParameters) -> Scalar:
        _ = hyperparameters  # unused
        return jstats.gamma.logpdf(1 / parameters.std_log_var, 10, scale=1 / 10)


class StochVolParamPrior(ParameterPrior[LogVolRW, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LogVolRW:
        std_key, mr_key, ltv_key = jrandom.split(key, 3)

        std_mean = jnp.array(3.0)
        std_scale = jnp.array(1.5)
        std_lower = (0.0 - std_mean) / std_scale
        std_log_vol = std_mean + std_scale * jrandom.truncated_normal(
            std_key, lower=std_lower, upper=jnp.inf
        )

        mr_mean = jnp.array(10.0)
        mr_scale = jnp.array(10.0)
        mr_lower = (0.0 - mr_mean) / mr_scale
        mean_reversion = mr_mean + mr_scale * jrandom.truncated_normal(
            mr_key, lower=mr_lower, upper=jnp.inf
        )

        long_term_vol = jnp.exp(
            jnp.array(-2.0) + jnp.array(0.5) * jrandom.normal(ltv_key)
        )

        _ = hyperparameters  # unused

        return LogVolRW(
            std_log_vol=std_log_vol,
            mean_reversion=mean_reversion,
            long_term_vol=long_term_vol,
        )

    @staticmethod
    def log_prob(parameters: LogVolRW, hyperparameters: HyperParameters) -> Scalar:
        _ = hyperparameters  # unused
        mean = 3.0
        scale = 0.5
        x = parameters.std_log_vol
        alpha = -mean / scale
        normalization = 1 - jstats.norm.cdf(alpha)

        z = (x - mean) / scale
        log_numerator = jstats.norm.logpdf(z) - jnp.log(scale)
        log_denominator = jnp.log(normalization)
        std_log_vol_lpdf = jnp.where(
            (x >= 0.0),
            log_numerator - log_denominator,
            -jnp.inf,
        )

        base_log_lpdf = jstats.norm.logpdf(
            jnp.log(parameters.long_term_vol),
            loc=jnp.array(-2.0),
            scale=jnp.array(0.5),
        )

        mean = 10
        scale = 10.0
        x = parameters.mean_reversion
        alpha = -mean / scale
        normalization = 1 - jstats.norm.cdf(alpha)

        z = (x - mean) / scale
        log_numerator = jstats.norm.logpdf(z) - jnp.log(scale)
        log_denominator = jnp.log(normalization)
        mean_reversion_lpdf = jnp.where(
            (x >= 0.0),
            log_numerator - log_denominator,
            -jnp.inf,
        )

        return std_log_vol_lpdf + base_log_lpdf + mean_reversion_lpdf


class StdLogVolPrior(ParameterPrior[LVolStd, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LVolStd:
        base_p = StochVolParamPrior().sample(key, hyperparameters)

        _ = hyperparameters  # unused

        return LVolStd(
            std_log_vol=base_p.std_log_vol,
            # long_term_vol=base_p.long_term_vol,
        )

    @staticmethod
    def log_prob(parameters: LVolStd, hyperparameters: HyperParameters) -> Scalar:
        _ = hyperparameters  # unused
        mean = 3.0
        scale = 0.5
        x = parameters.std_log_vol
        alpha = -mean / scale
        normalization = 1 - jstats.norm.cdf(alpha)

        z = (x - mean) / scale
        log_numerator = jstats.norm.logpdf(z) - jnp.log(scale)
        log_denominator = jnp.log(normalization)
        std_log_vol_lpdf = jnp.where(
            (x >= 0.0),
            log_numerator - log_denominator,
            -jnp.inf,
        )

        # base_log_lpdf = jstats.norm.logpdf(
        #     jnp.log(parameters.long_term_vol),
        #     loc=jnp.array(-2.0),
        #     scale=jnp.array(0.5),
        # )
        return std_log_vol_lpdf


class SkewStochVolParamPrior(ParameterPrior[LogVolWithSkew, HyperParameters]):
    @staticmethod
    def _base_parameters(parameters: LogVolWithSkew) -> LogVolRW:
        return LogVolRW(
            std_log_vol=parameters.std_log_vol,
            mean_reversion=parameters.mean_reversion,
            long_term_vol=parameters.long_term_vol,
        )

    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LogVolWithSkew:
        base_key, skew_key = jrandom.split(key)
        base_parameters = StochVolParamPrior.sample(base_key, hyperparameters)

        skew = jrandom.uniform(
            skew_key,
            shape=(),
            minval=jnp.array(-1.0, dtype=jnp.float32),
            maxval=jnp.array(1.0, dtype=jnp.float32),
        )

        return LogVolWithSkew(
            std_log_vol=base_parameters.std_log_vol,
            mean_reversion=base_parameters.mean_reversion,
            long_term_vol=base_parameters.long_term_vol,
            skew=skew,
        )

    @staticmethod
    def log_prob(
        parameters: LogVolWithSkew, hyperparameters: HyperParameters
    ) -> Scalar:
        base_parameters = SkewStochVolParamPrior._base_parameters(parameters)
        base_log_prob = StochVolParamPrior.log_prob(base_parameters, hyperparameters)

        skew = parameters.skew
        skew_log_prob = jnp.where(
            (skew >= -1.0) & (skew <= 1.0),
            -jnp.log(2.0),
            -jnp.inf,
        )

        return base_log_prob + skew_log_prob


"""
Latent Priors
"""


def gaussian_var_start_sample(
    key: PRNGKeyArray,
    conditions: tuple[NoCondition],
    parameters: LogVarAR,
) -> tuple[LatentVar]:
    # mu = 2 * jnp.log(0.16)
    # sigma = jnp.array(0.5)
    mu = jnp.array(0.0)
    sigma = jnp.sqrt(jnp.square(parameters.std_log_var) / (1 - jnp.square(parameters.ar)))
    start_lv = LatentVar(log_var=mu + sigma * jrandom.normal(key))
    return (start_lv,)


def gaussian_var_start_log_prob(
    latent: tuple[LatentVar],
    conditions: tuple[NoCondition],
    parameters: LogVarAR,
) -> Scalar:
    (start_lv,) = latent
    # mu = 2 * jnp.log(0.16)
    # sigma = jnp.array(0.5)
    mu = jnp.array(0.0)
    sigma = jnp.sqrt(jnp.square(parameters.std_log_var) / (1 - jnp.square(parameters.ar)))

    base_log_p = jstats.norm.logpdf(
        start_lv.log_var,
        loc=mu,
        scale=sigma,
    )
    return base_log_p


gaussian_var_start = Prior[tuple[LatentVar], tuple[NoCondition], LogVarAR](
    order=1, sample=gaussian_var_start_sample, log_prob=gaussian_var_start_log_prob
)


class _RandomWalkParameters(Protocol):
    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar


def _random_walk_loc_scale(
    prev_latent: LatentVol,
    condition: TimeIncrement,
    parameters: _RandomWalkParameters,
) -> tuple[Scalar, Scalar]:
    move_scale = jnp.sqrt(condition.dt) * parameters.std_log_vol
    move_loc = prev_latent.log_vol + condition.dt * parameters.mean_reversion * (
        jnp.log(parameters.long_term_vol) - prev_latent.log_vol
    )
    return move_loc, move_scale


def gaussian_start_sample(
    key: PRNGKeyArray,
    conditions: tuple[TimeIncrement],
    parameters: LogVolRW,
) -> tuple[LatentVol]:
    mu = jnp.log(jnp.array(0.1))
    sigma = jnp.array(1.6) / jnp.sqrt(2.0 * 6.0)

    start_lv = LatentVol(log_vol=(mu + sigma * jrandom.normal(key)))
    return (start_lv,)


def gaussian_start_log_prob(
    latent: tuple[LatentVol],
    conditions: tuple[TimeIncrement],
    parameters: LogVolRW,
) -> Scalar:
    (start_lv,) = latent
    mu = jnp.log(jnp.array(0.1))
    sigma = jnp.array(1.6) / jnp.sqrt(2.0 * 6.0)
    base_log_p = jstats.norm.logpdf(
        start_lv.log_vol,
        loc=mu,
        scale=sigma,
    )
    return base_log_p


gaussian_start = Prior[tuple[LatentVol], tuple[TimeIncrement], LogVolRW](
    order=1, sample=gaussian_start_sample, log_prob=gaussian_start_log_prob
)


def gaussian_two_step_sample(
    key: PRNGKeyArray,
    conditions: tuple[TimeIncrement, TimeIncrement],
    parameters: LogVolWithSkew,
) -> tuple[LatentVol, LatentVol]:
    mu = jnp.array(-2.0)
    sigma = jnp.array(0.5)

    start_key, trans_key = jrandom.split(key)
    start_lv = LatentVol(log_vol=mu + sigma * jrandom.normal(start_key))
    loc, scale = _random_walk_loc_scale(start_lv, conditions[1], parameters)
    next_lv = LatentVol(log_vol=loc + scale * jrandom.normal(trans_key))
    return start_lv, next_lv


def gaussian_two_step_log_prob(
    latent: tuple[LatentVol, LatentVol],
    conditions: tuple[TimeIncrement, TimeIncrement],
    parameters: LogVolWithSkew,
) -> Scalar:
    start_lv, next_lv = latent
    mu = jnp.array(-2.0)
    sigma = jnp.array(0.5)

    base_log_p = jstats.norm.logpdf(
        start_lv.log_vol,
        loc=mu,
        scale=sigma,
    )
    loc, scale = _random_walk_loc_scale(start_lv, conditions[1], parameters)
    log_vol = jnp.clip(next_lv.log_vol, a_min=jnp.log(0.001), a_max=jnp.log(10))
    rw_log_p = jstats.norm.logpdf(log_vol, loc=loc, scale=scale)
    return base_log_p + rw_log_p


two_step_gaussian_start = Prior[
    tuple[LatentVol, LatentVol],
    tuple[TimeIncrement, TimeIncrement],
    LogVolWithSkew,
](
    order=2,
    sample=gaussian_two_step_sample,
    log_prob=gaussian_two_step_log_prob,
)


"""
Transitions
"""


def loc_scale(
    latent_history: tuple[LatentVol],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> tuple[Scalar, Scalar]:
    (prev_latent,) = latent_history
    return _random_walk_loc_scale(prev_latent, condition, parameters)


random_walk: Transition[
    tuple[LatentVol],
    LatentVol,
    TimeIncrement,
    LogVolRW,
] = GaussianLocScaleTransition(
    loc_scale=loc_scale,
    latent_t=LatentVol,
)


def random_walk_ar_sample(
    key: PRNGKeyArray,
    latent_history: tuple[LatentVar],
    condition: NoCondition,
    parameters: LogVarAR,
) -> LatentVar:
    (last_log_var,) = latent_history
    loc = parameters.ar * last_log_var.log_var
    scale = parameters.std_log_var
    return LatentVar(log_var=loc + scale * jrandom.normal(key))


def random_walk_ar_log_prob(
    latent_history: tuple[LatentVar],
    latent: LatentVar,
    condition: NoCondition,
    parameters: LogVarAR,
) -> Scalar:
    (last_log_var,) = latent_history
    loc = parameters.ar * last_log_var.log_var
    scale = parameters.std_log_var
    return jstats.norm.logpdf(latent.log_var, loc=loc, scale=scale)


random_walk_ar = Transition[tuple[LatentVar], LatentVar, NoCondition, LogVarAR](
    order=1,
    sample=random_walk_ar_sample,
    log_prob=random_walk_ar_log_prob,
)


def skew_rw_loc_scale(
    latent_history: tuple[LatentVol, LatentVol],
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> tuple[Scalar, Scalar]:
    last_latent = latent_history[-1]
    return _random_walk_loc_scale(last_latent, condition, parameters)


def skew_rw_sample(
    key: PRNGKeyArray,
    latent_history: tuple[LatentVol, LatentVol],
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> LatentVol:
    loc, scale = skew_rw_loc_scale(latent_history, condition, parameters)
    return LatentVol(log_vol=loc + scale * jrandom.normal(key))


def skew_rw_log_prob(
    latent_history: tuple[LatentVol, LatentVol],
    latent: LatentVol,
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> Scalar:
    loc, scale = skew_rw_loc_scale(latent_history, condition, parameters)
    log_vol = jnp.clip(latent.log_vol, a_min=jnp.log(0.001), a_max=jnp.log(10))
    return jstats.norm.logpdf(log_vol, loc=loc, scale=scale)


skew_random_walk = Transition[
    tuple[LatentVol, LatentVol], LatentVol, TimeIncrement, LogVolWithSkew
](
    order=2,
    sample=skew_rw_sample,
    log_prob=skew_rw_log_prob,
)


def skew_return_mean_and_scale(
    last_latent: LatentVol,
    current_latent: LatentVol,
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> tuple[Scalar, Scalar]:
    dt = condition.dt

    current_vol = jnp.exp(current_latent.log_vol)
    current_var = jnp.exp(2 * current_latent.log_vol)

    log_vol_mean, _ = _random_walk_loc_scale(last_latent, condition, parameters)

    return_mean = -0.5 * dt * current_var
    return_mean += (
        parameters.skew
        * (current_vol / parameters.std_log_vol)
        * (current_latent.log_vol - log_vol_mean)
    )

    return_scale = (
        jnp.sqrt(condition.dt) * current_vol * jnp.sqrt(1 - parameters.skew**2)
    )

    return return_mean, return_scale


"""
Emissions
"""


def log_return_var_sample(
    key: PRNGKeyArray,
    latent: tuple[LatentVar],
    observation_history: tuple[()],
    condition: NoCondition,
    parameters: LogVarAR,
) -> LogReturnObs:
    (current_latent,) = latent
    return_scale = jnp.exp(0.5 * current_latent.log_var)
    log_return = jrandom.normal(key) * return_scale
    return LogReturnObs(log_return=log_return)


def log_return_var_prob(
    latent: tuple[LatentVar],
    observation: LogReturnObs,
    observation_history: tuple[()],
    condition: NoCondition,
    parameters: LogVarAR,
) -> Scalar:
    (current_latent,) = latent
    return_scale = jnp.exp(0.5 * current_latent.log_var)
    log_return = observation.log_return
    return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)


log_return_var = Emission[tuple[LatentVar], NoCondition, LogReturnObs, LogVarAR](
    order=1,
    sample=log_return_var_sample,
    log_prob=log_return_var_prob,
)


def log_return_sample(
    key: PRNGKeyArray,
    latent: tuple[LatentVol],
    observation_history: tuple[()],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> LogReturnObs:
    (current_latent,) = latent
    return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_latent.log_vol)
    log_return = jrandom.normal(key) * return_scale
    return LogReturnObs(log_return=log_return)


def log_return_log_prob(
    latent: tuple[LatentVol],
    observation: LogReturnObs,
    observation_history: tuple[()],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> Scalar:
    (current_latent,) = latent
    return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_latent.log_vol)
    log_return = observation.log_return
    return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)


log_return = Emission[tuple[LatentVol], TimeIncrement, LogReturnObs, LogVolRW](
    order=1,
    sample=log_return_sample,
    log_prob=log_return_log_prob,
)


def skew_sample(
    key: PRNGKeyArray,
    latent: tuple[LatentVol, LatentVol],
    observation_history: tuple[()],
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> LogReturnObs:
    last_latent, current_latent = latent
    return_mean, return_scale = skew_return_mean_and_scale(
        last_latent,
        current_latent,
        condition,
        parameters,
    )
    log_return = jrandom.normal(key) * return_scale + return_mean
    return LogReturnObs(log_return=log_return)


def skew_log_prob(
    latent: tuple[LatentVol, LatentVol],
    observation: LogReturnObs,
    observation_history: tuple[()],
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> Scalar:
    last_latent, current_latent = latent
    return_mean, return_scale = skew_return_mean_and_scale(
        last_latent,
        current_latent,
        condition,
        parameters,
    )
    log_return = observation.log_return

    return jstats.norm.logpdf(log_return, loc=return_mean, scale=return_scale)


skew_log_return = Emission[
    tuple[LatentVol, LatentVol],
    TimeIncrement,
    LogReturnObs,
    LogVolWithSkew,
](
    order=2,
    sample=skew_sample,
    log_prob=skew_log_prob,
)


class SimpleStochasticVol(
    SequentialModel[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolRW,
    ]
):
    latent_cls = LatentVol
    observation_cls = LogReturnObs
    condition_cls = TimeIncrement
    parameter_cls = LogVolRW

    prior = gaussian_start
    transition = random_walk
    emission = log_return


class SimpleStochasticVolBayesianStdLogVol(
    BayesianSequentialModel[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolRW,
        LVolStd,
        HyperParameters,
    ]
):
    def __init__(self, ref_params: LogVolRW):
        self.target_parameter = staticmethod(
            partial(lv_to_std_only, ref_params=ref_params)
        )

    inference_parameter_cls = LVolStd
    target = SimpleStochasticVol()
    parameter_prior = StdLogVolPrior()


class SimpleStochasticVar(
    SequentialModel[
        LatentVar,
        LogReturnObs,
        NoCondition,
        LogVarAR,
    ]
):
    latent_cls = LatentVar
    observation_cls = LogReturnObs
    condition_cls = NoCondition
    parameter_cls = LogVarAR

    prior = gaussian_var_start
    transition = random_walk_ar
    emission = log_return_var


class SimpleStochasticVarBayesian(
    BayesianSequentialModel[
        LatentVar,
        LogReturnObs,
        NoCondition,
        LogVarAR,
        LogVarStd,
        HyperParameters,
    ]
):
    def __init__(self, ref_params: LogVarAR):
        self.target_parameter = staticmethod(
            partial(lvar_to_std_only, ref_params=ref_params)
        )

    inference_parameter_cls = LogVarStd
    target = SimpleStochasticVar()
    parameter_prior = StochVarPrior()


class SimpleStochasticVolBayesian(
    BayesianSequentialModel[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolRW,
        LogVolRW,
        HyperParameters,
    ]
):
    inference_parameter_cls = LogVolRW
    target = SimpleStochasticVol()
    parameter_prior = StochVolParamPrior()
    target_parameter = staticmethod(lambda parameters: parameters)


class SkewStochasticVol(
    SequentialModel_TO2_EO2[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolWithSkew,
    ]
):
    latent_cls = LatentVol
    condition_cls = TimeIncrement
    observation_cls = LogReturnObs
    parameter_cls = LogVolWithSkew

    prior = two_step_gaussian_start
    transition = skew_random_walk
    emission = skew_log_return


class SkewStochasticVolBayesian(
    BayesianSequentialModel_TO2_EO2[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolWithSkew,
        LogVolWithSkew,
        HyperParameters,
    ]
):
    inference_parameter_cls = LogVolWithSkew
    target = SkewStochasticVol()
    parameter_prior = SkewStochVolParamPrior()
    target_parameter = staticmethod(lambda parameters: parameters)


def make_constant_time_increments(
    sequence_length: int,
    *,
    dt: float = 1.0,
    prior_order: int | None = None,
) -> TimeIncrement:
    """Return a ``TimeIncrement`` tree filled with a constant ``dt``."""

    if sequence_length < 1:
        raise ValueError(
            f"sequence_length must be >= 1, got {sequence_length}",
        )

    effective_prior_order = (
        SimpleStochasticVol.prior.order if prior_order is None else prior_order
    )
    required_length = sequence_length + effective_prior_order - 1
    dt_value = jnp.asarray(dt, dtype=jnp.float32)
    increments = jnp.full((required_length,), dt_value, dtype=dt_value.dtype)
    return TimeIncrement(dt=increments)
