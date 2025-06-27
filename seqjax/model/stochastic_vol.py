from typing import Union, ClassVar, Literal
from dataclasses import dataclass, field

from jaxtyping import Scalar, PRNGKeyArray
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.random as jrandom
import haliax as hax

from seqjax.model.typing import (
    Particle,
    Observation,
    Condition,
    Parameters,
    HyperParameters,
)
from seqjax.model.base import (
    Target,
    Transition,
    Prior,
    Emission,
    ParameterPrior,
)

"""
All values are in annualised terms.
"""


# Latent Particles
class LatentVol(Particle):
    log_vol: Scalar
    _spec_log_vol = ()


class Underlying(Observation):
    underlying: Scalar
    _spec_underlying = ()


# parameters
class LogVolRW(Parameters):
    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar

    # initial values
    reference_emission: tuple[Underlying] = (Underlying(jnp.array(3000.0)),)


class LogVolWithSkew(Parameters):
    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar
    skew: Scalar  # correlation between random variations

    # initial values
    reference_emission: tuple[Underlying] = (Underlying(jnp.array(3000.0)),)


LogVolRandomWalks = Union[LogVolRW, LogVolWithSkew]


class TimeIncrement(Condition):
    dt: Scalar  # time since last observation
    _spec_dt = ()


class SotchVolParamPrior(ParameterPrior[LogVolRW, HyperParameters]):
    @staticmethod
    def sample(key, hyperparameters):
        pass

    @staticmethod
    def log_p(parameteters, hyperparameters=None):
        mean = 3.0
        scale = 1.0
        x = parameteters.std_log_vol
        alpha = -mean / scale
        normalization = 1 - jstats.norm.cdf(alpha)

        z = (x - mean) / scale
        log_numerator = jstats.norm.logpdf(z) - jnp.log(scale)
        log_denominator = hax.log(normalization)
        std_log_vol_lpdf = hax.where(
            (x >= 0.0), log_numerator - log_denominator, -hax.inf
        )

        base_log_lpdf = jstats.norm.logpdf(
            hax.log(parameteters.long_term_vol),
            loc=jnp.array(-2.0),
            scale=jnp.array(0.5),
        )

        mean = 10
        scale = 10.0
        x = parameteters.mean_reversion
        alpha = -mean / scale
        normalization = 1 - jstats.norm.cdf(alpha)

        z = (x - mean) / scale
        log_numerator = jstats.norm.logpdf(z) - hax.log(scale)
        log_denominator = hax.log(normalization)
        mean_reversion_lpdf = hax.where(
            (x >= 0.0), log_numerator - log_denominator, -hax.inf
        )

        return std_log_vol_lpdf + base_log_lpdf + mean_reversion_lpdf


class GaussianStart(Prior[LatentVol, TimeIncrement, LogVolRandomWalks]):
    order: ClassVar[int] = 2

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[TimeIncrement, TimeIncrement],
        parameters: LogVolRandomWalks,
    ) -> tuple[LatentVol, LatentVol]:
        mu = jnp.array(-2.0)
        sigma = jnp.array(0.5)

        start_key, trans_key = jrandom.split(key)
        start_lv = LatentVol(log_vol=mu + sigma * jrandom.normal(start_key))
        next_lv = RandomWalk.sample(
            trans_key,
            (start_lv,),
            conditions[1],
            parameters,
        )
        return start_lv, next_lv

    @staticmethod
    def log_p(
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
        rw_log_p = RandomWalk.log_p(
            (start_lv,),
            next_lv,
            conditions[1],
            parameters,
        )
        return base_log_p + rw_log_p


class RandomWalk(Transition[LatentVol, TimeIncrement, LogVolRandomWalks]):
    order: ClassVar[int] = 1

    @staticmethod
    def loc_scale(
        particle_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ):
        move_scale = hax.sqrt(condition.dt) * parameters.std_log_vol
        (prev_particle,) = particle_history  # unpack

        move_loc = prev_particle.log_vol + condition.dt * parameters.mean_reversion * (
            hax.log(parameters.long_term_vol) - prev_particle.log_vol
        )

        return move_loc, move_scale

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ) -> LatentVol:
        loc, scale = RandomWalk.loc_scale(particle_history, condition, parameters)
        return LatentVol(loc + scale * jrandom.normal(key))

    @staticmethod
    def log_p(
        particle_history: tuple[LatentVol],
        particle: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ) -> Scalar:
        loc, scale = RandomWalk.loc_scale(particle_history, condition, parameters)
        # clip evaluations to reasonable level, if we are sampling outside this frequently
        # something has gone wrong
        # TODO: should this be an error instead?
        log_vol = hax.clip(particle.log_vol, a_min=jnp.log(0.001), a_max=jnp.log(10))
        return jstats.norm.logpdf(log_vol, loc=loc, scale=scale)


class LogReturn(Emission[LatentVol, Underlying, TimeIncrement, LogVolRW]):
    order: ClassVar[int] = 2  # depends on last particle and particle now
    observation_dependency: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[LatentVol, LatentVol],
        observation_history: tuple[Underlying],
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Underlying:
        last_particle, _ = particle
        (prev_observation,) = observation_history
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(last_particle.log_vol)
        log_return = jrandom.normal(key) * return_scale
        return Underlying(underlying=prev_observation.underlying * jnp.exp(log_return))

    @staticmethod
    def log_p(
        particle: tuple[LatentVol, LatentVol],
        observation_history: tuple[Underlying],
        observation: Underlying,
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Scalar:
        last_particle, _ = particle
        (prev_observation,) = observation_history
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(last_particle.log_vol)
        log_return = jnp.log(observation.underlying) - jnp.log(
            prev_observation.underlying
        )
        return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)


class SkewLogReturn(Emission[LatentVol, Underlying, TimeIncrement, LogVolWithSkew]):
    order: ClassVar[int] = 2  # depends on last particle and particle now
    observation_dependency: ClassVar[int] = 1

    @staticmethod
    def return_mean_and_scale(
        last_particle: LatentVol,
        current_particle: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> tuple[Scalar, Scalar]:
        dt = condition.dt

        last_vol = jnp.exp(last_particle.log_vol)
        last_var = jnp.exp(2 * last_particle.log_vol)

        log_vol_mean, _ = RandomWalk.loc_scale(
            (last_particle,),
            condition,
            parameters,
        )

        return_mean = -0.5 * dt * last_var
        return_mean += (
            parameters.skew
            * (last_vol / parameters.std_log_vol)
            * (current_particle.log_vol - log_vol_mean)
        )

        return_scale = (
            jnp.sqrt(condition.dt) * last_vol * jnp.sqrt(1 - parameters.skew**2)
        )

        return return_mean, return_scale

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[LatentVol, LatentVol],
        observation_history: tuple[Underlying],
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> Underlying:
        last_particle, current_particle = particle
        (prev_observation,) = observation_history
        return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
            last_particle, current_particle, condition, parameters
        )
        log_return = jrandom.normal(key) * return_scale + return_mean
        return Underlying(underlying=prev_observation.underlying * jnp.exp(log_return))

    @staticmethod
    def log_p(
        particle: tuple[LatentVol, LatentVol],
        observation_history: tuple[Underlying],
        observation: Underlying,
        condition: TimeIncrement,
        parameters: LogVolWithSkew,
    ) -> Scalar:
        last_particle, current_particle = particle
        (prev_observation,) = observation_history
        return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
            last_particle, current_particle, condition, parameters
        )

        log_return = jnp.log(observation.underlying) - jnp.log(
            prev_observation.underlying
        )

        return jstats.norm.logpdf(log_return, loc=return_mean, scale=return_scale)


class SimpleStochasticVol(Target[LatentVol, Underlying, TimeIncrement, LogVolRW]):
    prior = GaussianStart()
    transition = RandomWalk()
    emission = LogReturn()


class SkewStochasticVol(Target[LatentVol, Underlying, TimeIncrement, LogVolWithSkew]):
    prior = GaussianStart()
    transition = RandomWalk()
    emission = SkewLogReturn()


@dataclass
class StochasticVolConfig:
    label: Literal["simple_stochastic_vol"] = field(
        init=False, default="simple_stochastic_vol"
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
