"""Stochastic volatility models."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Union

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    Emission,
    ParameterPrior,
    Prior,
    SequentialModel,
    Transition,
)
from seqjax.model.typing import (
    Condition,
    HyperParameters,
    Observation,
    Parameters,
    Particle,
)

"""
All values are in annualised terms.
"""


# Latent Particles
class LatentVol(Particle):
    """Latent state containing the log volatility."""

    log_vol: Scalar


class LogReturnObs(Observation):
    """Observed log return."""

    log_return: Scalar


# parameters
class LogVolRW(Parameters):
    """Parameters for a log-volatility random walk."""

    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar

    # initial values
    reference_emission: tuple[LogReturnObs] = field(default_factory=tuple)


class LogVolWithSkew(Parameters):
    """Random-walk parameters including a skew term."""

    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar
    skew: Scalar  # correlation between random variations

    # initial values
    reference_emission: tuple[LogReturnObs] = field(default_factory=tuple)


LogVolRandomWalks = Union[LogVolRW, LogVolWithSkew]


class TimeIncrement(Condition):
    """Time step between observations."""

    dt: Scalar  # time since last observation


class StochVolParamPrior(ParameterPrior[LogVolRW, HyperParameters]):
    @staticmethod
    def sample(key, hyperparameters):
        std_key, mr_key, ltv_key = jrandom.split(key, 3)

        std_mean = jnp.array(3.0)
        std_scale = jnp.array(1.0)
        std_lower = (0.0 - std_mean) / std_scale
        std_log_vol = (
            std_mean
            + std_scale
            * jrandom.truncated_normal(std_key, lower=std_lower, upper=jnp.inf)
        )

        mr_mean = jnp.array(10.0)
        mr_scale = jnp.array(10.0)
        mr_lower = (0.0 - mr_mean) / mr_scale
        mean_reversion = (
            mr_mean
            + mr_scale
            * jrandom.truncated_normal(mr_key, lower=mr_lower, upper=jnp.inf)
        )

        long_term_vol = jnp.exp(
            jnp.array(-2.0) + jnp.array(0.5) * jrandom.normal(ltv_key)
        )

        return LogVolRW(
            std_log_vol=std_log_vol,
            mean_reversion=mean_reversion,
            long_term_vol=long_term_vol,
        )

    @staticmethod
    def log_prob(parameters, hyperparameters=None):
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


class GaussianStart(Prior[LatentVol, TimeIncrement, LogVolRandomWalks]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(  # type: ignore[override]
        key: PRNGKeyArray,
        conditions: tuple[TimeIncrement],
        parameters: LogVolRandomWalks,
    ) -> tuple[LatentVol]:
        mu = jnp.array(-2.0)
        sigma = jnp.array(0.5)

        start_lv = LatentVol(log_vol=mu + sigma * jrandom.normal(key))
        return (start_lv,)

    @staticmethod
    def log_prob(  # type: ignore[override]
        particle: tuple[LatentVol],
        conditions: tuple[TimeIncrement],
        parameters: LogVolRandomWalks,
    ) -> Scalar:
        (start_lv,) = particle
        mu = jnp.array(-2.0)
        sigma = jnp.array(0.5)

        base_log_p = jstats.norm.logpdf(
            start_lv.log_vol,
            loc=mu,
            scale=sigma,
        )
        return base_log_p


class TwoStepGaussianStart(Prior[LatentVol, TimeIncrement, LogVolRandomWalks]):
    """Prior that also samples the first transition step."""

    order: ClassVar[int] = 2

    @staticmethod
    def sample(  # type: ignore[override]
        key: PRNGKeyArray,
        conditions: tuple[TimeIncrement, TimeIncrement],
        parameters: LogVolRandomWalks,
    ) -> tuple[LatentVol, LatentVol]:
        mu = jnp.array(-2.0)
        sigma = jnp.array(0.5)

        start_key, trans_key = jrandom.split(key)
        start_lv = LatentVol(log_vol=mu + sigma * jrandom.normal(start_key))
        next_lv = RandomWalk.sample(trans_key, (start_lv,), conditions[1], parameters)
        return start_lv, next_lv

    @staticmethod
    def log_prob(  # type: ignore[override]
        particle: tuple[LatentVol, LatentVol],
        conditions: tuple[TimeIncrement, TimeIncrement],
        parameters: LogVolRandomWalks,
    ) -> Scalar:
        start_lv, next_lv = particle
        mu = jnp.array(-2.0)
        sigma = jnp.array(0.5)

        base_log_p = jstats.norm.logpdf(
            start_lv.log_vol,
            loc=mu,
            scale=sigma,
        )
        rw_log_p = RandomWalk.log_prob((start_lv,), next_lv, conditions[1], parameters)
        return base_log_p + rw_log_p


class RandomWalk(Transition[LatentVol, TimeIncrement, LogVolRandomWalks]):
    order: ClassVar[int] = 1

    @staticmethod
    def loc_scale(  # type: ignore[override]
        particle_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ):
        move_scale = jnp.sqrt(condition.dt) * parameters.std_log_vol
        (prev_particle,) = particle_history  # unpack

        move_loc = prev_particle.log_vol + condition.dt * parameters.mean_reversion * (
            jnp.log(parameters.long_term_vol) - prev_particle.log_vol
        )

        return move_loc, move_scale

    @staticmethod
    def sample(  # type: ignore[override]
        key: PRNGKeyArray,
        particle_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ) -> LatentVol:
        loc, scale = RandomWalk.loc_scale(particle_history, condition, parameters)
        return LatentVol(loc + scale * jrandom.normal(key))

    @staticmethod
    def log_prob(  # type: ignore[override]
        particle_history: tuple[LatentVol],
        particle: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ) -> Scalar:
        loc, scale = RandomWalk.loc_scale(particle_history, condition, parameters)
        # clip evaluations to reasonable level, if we are sampling outside this frequently
        # something has gone wrong
        # TODO: should this be an error instead?
        log_vol = jnp.clip(particle.log_vol, a_min=jnp.log(0.001), a_max=jnp.log(10))
        return jstats.norm.logpdf(log_vol, loc=loc, scale=scale)


class LogReturn(Emission[LatentVol, LogReturnObs, TimeIncrement, LogVolRW]):
    order: ClassVar[int] = 1  # depends only on current particle
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(  # type: ignore[override]
        key: PRNGKeyArray,
        particle: tuple[LatentVol],
        observation_history: tuple[()],
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> LogReturnObs:
        (current_particle,) = particle
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_particle.log_vol)
        log_return = jrandom.normal(key) * return_scale
        return LogReturnObs(log_return=log_return)

    @staticmethod
    def log_prob(  # type: ignore[override]
        particle: tuple[LatentVol],
        observation_history: tuple[()],
        observation: LogReturnObs,
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Scalar:
        (current_particle,) = particle
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_particle.log_vol)
        log_return = observation.log_return
        return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)


class SkewLogReturn(Emission[LatentVol, LogReturnObs, TimeIncrement, LogVolWithSkew]):
    order: ClassVar[int] = 2  # depends on last particle and current particle
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def return_mean_and_scale(
        last_particle: LatentVol,
        current_particle: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> tuple[Scalar, Scalar]:
        dt = condition.dt

        current_vol = jnp.exp(current_particle.log_vol)
        current_var = jnp.exp(2 * current_particle.log_vol)

        log_vol_mean, _ = RandomWalk.loc_scale(
            (last_particle,),
            condition,
            parameters,
        )

        return_mean = -0.5 * dt * current_var
        return_mean += (
            parameters.skew
            * (current_vol / parameters.std_log_vol)
            * (current_particle.log_vol - log_vol_mean)
        )

        return_scale = (
            jnp.sqrt(condition.dt) * current_vol * jnp.sqrt(1 - parameters.skew**2)
        )

        return return_mean, return_scale

    @staticmethod
    def sample(  # type: ignore[override]
        key: PRNGKeyArray,
        particle: tuple[LatentVol, LatentVol],
        observation_history: tuple[()],
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> LogReturnObs:
        last_particle, current_particle = particle
        return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
            last_particle,
            current_particle,
            condition,
            parameters,
        )
        _ = observation_history  # unused
        log_return = jrandom.normal(key) * return_scale + return_mean
        return LogReturnObs(log_return=log_return)

    @staticmethod
    def log_prob(  # type: ignore[override]
        particle: tuple[LatentVol, LatentVol],
        observation_history: tuple[()],
        observation: LogReturnObs,
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> Scalar:
        last_particle, current_particle = particle
        return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
            last_particle,
            current_particle,
            condition,
            parameters,
        )

        _ = observation_history  # unused
        log_return = observation.log_return

        return jstats.norm.logpdf(log_return, loc=return_mean, scale=return_scale)


class SimpleStochasticVol(SequentialModel[LatentVol, LogReturnObs, TimeIncrement, LogVolRW]):
    prior = GaussianStart()
    transition = RandomWalk()
    emission = LogReturn()


class SkewStochasticVol(SequentialModel[LatentVol, LogReturnObs, TimeIncrement, LogVolWithSkew]):
    prior = TwoStepGaussianStart()
    transition = RandomWalk()
    emission = SkewLogReturn()


@dataclass
class StochasticVolConfig:
    """Configuration for generating stochastic volatility data."""
    label: Literal["simple_stochastic_vol"] = field(
        init=False,
        default="simple_stochastic_vol",
    )
    path_length: int
    data_seed: int

    @property
    def data_key(self):
        return jrandom.split(jrandom.key(self.data_seed))[0]

    @property
    def hyperparam_key(self):
        return jrandom.split(jrandom.key(self.data_seed))[1]

    @property
    def generative_parameters(self) -> LogVolRW:
        return LogVolRW(
            std_log_vol=jnp.array(3.2),
            mean_reversion=jnp.array(12.0),
            long_term_vol=jnp.array(0.16),
        )

    @property
    def hyperparameters(self) -> HyperParameters:
        return HyperParameters()
