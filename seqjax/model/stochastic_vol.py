"""Stochastic volatility models.
All values are in annualised terms.
"""

from collections import OrderedDict
from typing import ClassVar, Protocol

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    BayesianSequentialModel,
    EmissionO1D0,
    EmissionO2D0,
    ParameterPrior,
    Prior1,
    Prior2,
    SequentialModel,
    Transition1,
    Transition2,
)
from seqjax.model.typing import (
    Condition,
    HyperParameters,
    Observation,
    Parameters,
    Latent,
)


class LatentVol(Latent):
    """Latent state containing the log volatility."""

    log_vol: Scalar
    _shape_template: ClassVar = OrderedDict(
        log_vol=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
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


class TimeIncrement(Condition):
    """Time step between observations."""

    dt: Scalar  # time since last observation
    _shape_template: ClassVar = OrderedDict(
        dt=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class StochVolParamPrior(ParameterPrior[LogVolRW, HyperParameters]):
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: HyperParameters) -> LogVolRW:
        std_key, mr_key, ltv_key = jrandom.split(key, 3)

        std_mean = jnp.array(3.0)
        std_scale = jnp.array(1.0)
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
        scale = 1.0
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


class GaussianStart(Prior1[LatentVol, TimeIncrement, LogVolRW]):
    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: TimeIncrement,
        parameters: LogVolRW,
    ) -> tuple[LatentVol]:
        mu = jnp.array(-2.0)
        sigma = jnp.array(0.5)

        start_lv = LatentVol(log_vol=mu + sigma * jrandom.normal(key))
        return (start_lv,)

    @staticmethod
    def log_prob(
        latent: tuple[LatentVol],
        conditions: TimeIncrement,
        parameters: LogVolRW,
    ) -> Scalar:
        (start_lv,) = latent
        mu = jnp.array(-2.0)
        sigma = jnp.array(0.5)

        base_log_p = jstats.norm.logpdf(
            start_lv.log_vol,
            loc=mu,
            scale=sigma,
        )
        return base_log_p


class TwoStepGaussianStart(Prior2[LatentVol, TimeIncrement, LogVolWithSkew]):
    """Prior that also samples the first transition step."""

    @staticmethod
    def sample(
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

    @staticmethod
    def log_prob(
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


class RandomWalk(Transition1[LatentVol, TimeIncrement, LogVolRW]):
    @staticmethod
    def loc_scale(
        latent_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> tuple[Scalar, Scalar]:
        (prev_latent,) = latent_history
        return _random_walk_loc_scale(prev_latent, condition, parameters)

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        latent_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> LatentVol:
        loc, scale = RandomWalk.loc_scale(latent_history, condition, parameters)
        return LatentVol(log_vol=loc + scale * jrandom.normal(key))

    @staticmethod
    def log_prob(
        latent_history: tuple[LatentVol],
        latent: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Scalar:
        loc, scale = RandomWalk.loc_scale(latent_history, condition, parameters)
        # clip evaluations to reasonable level, if we are sampling outside this frequently
        # something has gone wrong
        # TODO: should this be an error instead?
        log_vol = jnp.clip(latent.log_vol, a_min=jnp.log(0.001), a_max=jnp.log(10))
        return jstats.norm.logpdf(log_vol, loc=loc, scale=scale)


class SkewRandomWalk(Transition2[LatentVol, TimeIncrement, LogVolWithSkew]):
    @staticmethod
    def loc_scale(
        latent_history: tuple[LatentVol, LatentVol],
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> tuple[Scalar, Scalar]:
        last_latent = latent_history[-1]
        return _random_walk_loc_scale(last_latent, condition, parameters)

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        latent_history: tuple[LatentVol, LatentVol],
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> LatentVol:
        loc, scale = SkewRandomWalk.loc_scale(latent_history, condition, parameters)
        return LatentVol(log_vol=loc + scale * jrandom.normal(key))

    @staticmethod
    def log_prob(
        latent_history: tuple[LatentVol, LatentVol],
        latent: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> Scalar:
        loc, scale = SkewRandomWalk.loc_scale(latent_history, condition, parameters)
        log_vol = jnp.clip(latent.log_vol, a_min=jnp.log(0.001), a_max=jnp.log(10))
        return jstats.norm.logpdf(log_vol, loc=loc, scale=scale)


class LogReturn(EmissionO1D0[LatentVol, LogReturnObs, TimeIncrement, LogVolRW]):
    @staticmethod
    def sample(
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

    @staticmethod
    def log_prob(
        latent: tuple[LatentVol],
        observation_history: tuple[()],
        observation: LogReturnObs,
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Scalar:
        (current_latent,) = latent
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_latent.log_vol)
        log_return = observation.log_return
        return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)


class SkewLogReturn(
    EmissionO2D0[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolWithSkew,
    ]
):
    @staticmethod
    def return_mean_and_scale(
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

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        latent: tuple[LatentVol, LatentVol],
        observation_history: tuple[()],
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> LogReturnObs:
        last_latent, current_latent = latent
        return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
            last_latent,
            current_latent,
            condition,
            parameters,
        )
        _ = observation_history  # unused
        log_return = jrandom.normal(key) * return_scale + return_mean
        return LogReturnObs(log_return=log_return)

    @staticmethod
    def log_prob(
        latent: tuple[LatentVol, LatentVol],
        observation_history: tuple[()],
        observation: LogReturnObs,
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> Scalar:
        last_latent, current_latent = latent
        return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
            last_latent,
            current_latent,
            condition,
            parameters,
        )

        _ = observation_history  # unused
        log_return = observation.log_return

        return jstats.norm.logpdf(log_return, loc=return_mean, scale=return_scale)


class SimpleStochasticVol(
    SequentialModel[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolRW,
    ]
):
    latent_cls: type[LatentVol] = LatentVol
    observation_cls: type[LogReturnObs] = LogReturnObs
    condition_cls: type[TimeIncrement] = TimeIncrement
    parameter_cls: type[LogVolRW] = LogVolRW

    prior = GaussianStart()
    transition = RandomWalk()
    emission = LogReturn()


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
    SequentialModel[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolWithSkew,
    ]
):
    latent_cls: type[LatentVol] = LatentVol
    condition_cls: type[TimeIncrement] = TimeIncrement
    observation_cls: type[LogReturnObs] = LogReturnObs
    parameter_cls: type[LogVolWithSkew] = LogVolWithSkew

    prior = TwoStepGaussianStart()
    transition = SkewRandomWalk()
    emission = SkewLogReturn()


class SkewStochasticVolBayesian(
    BayesianSequentialModel[
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
