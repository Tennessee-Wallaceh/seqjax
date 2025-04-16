from dataclasses import field

from jaxtyping import Scalar, PRNGKeyArray
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.random as jrandom

from seqjax.target.base import (
    Target,
    Particle,
    Observation,
    Condition,
    Parameters,
    Transition,
    Prior,
    Emission,
)

"""
All values are in annualised terms.
"""


# Latent Particles
class LatentVol(Particle):
    last_log_vol: Scalar
    current_log_vol: Scalar


# hyper parameters
class LogVolRW(Parameters):
    std_log_vol: Scalar

    # initial values
    initial_underlying: Scalar
    initial_min_vol: Scalar
    initial_max_vol: Scalar


class LogVolWithSkew(Parameters):
    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_log_vol: Scalar
    return_drift: Scalar
    skew: Scalar  # correlation between random variations

    # initial values
    initial_underlying: Scalar
    initial_min_vol: Scalar
    initial_max_vol: Scalar


LogVolRandomWalks = LogVolRW | LogVolWithSkew


class Underlying(Observation):
    underlying: Scalar


class LastUnderlying(Condition):
    dt: Scalar  # time since last observation
    last_underlying: Scalar = field(
        default_factory=lambda: jnp.array(jnp.nan)
    )  # fill in with nan when not available


class UniformStart(Prior[LatentVol, LogVolRandomWalks]):
    @staticmethod
    def sample(key: PRNGKeyArray, parameters: LogVolRandomWalks) -> LatentVol:
        vol = jrandom.uniform(
            key,
            minval=parameters.initial_min_vol,
            maxval=parameters.initial_max_vol,
        )
        return LatentVol(last_log_vol=jnp.log(vol), current_log_vol=jnp.log(vol))

    @staticmethod
    def log_p(particle: LatentVol, parameters: LogVolRandomWalks) -> Scalar:
        # re-parameterization
        uniform_width = parameters.initial_max_vol - parameters.initial_min_vol
        base_log_p = jstats.uniform.logpdf(
            jnp.exp(particle.last_log_vol),
            loc=parameters.initial_min_vol,
            scale=uniform_width,
        )
        log_jac_term = particle.last_log_vol
        return base_log_p + log_jac_term


class RandomWalk(Transition[LatentVol, LastUnderlying, LogVolRW]):
    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: LatentVol,
        condition: LastUnderlying,
        parameters: LogVolRW,
    ) -> LatentVol:
        move_scale = jnp.sqrt(condition.dt) * parameters.std_log_vol
        adjustment = -0.5 * condition.dt * parameters.std_log_vol**2
        next_log_vol = (
            particle.current_log_vol + adjustment + jrandom.normal(key) * move_scale
        )
        return LatentVol(
            current_log_vol=next_log_vol, last_log_vol=particle.current_log_vol
        )

    @staticmethod
    def log_p(
        particle: LatentVol,
        next_particle: LatentVol,
        condition: LastUnderlying,
        parameters: LogVolRW,
    ) -> Scalar:
        move_scale = jnp.sqrt(condition.dt) * parameters.std_log_vol
        adjustment = -0.5 * condition.dt * parameters.std_log_vol**2
        return jstats.norm.logpdf(
            next_particle.current_log_vol + adjustment,
            loc=particle.current_log_vol,
            scale=move_scale,
        ) + jnp.where(
            next_particle.last_log_vol == particle.current_log_vol,
            jnp.array(0.0),
            jnp.array(-jnp.inf),
        )  # delta mass on deterministic transition


class SkewRandomWalk(Transition[LatentVol, LastUnderlying, LogVolWithSkew]):
    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: LatentVol,
        condition: LastUnderlying,
        parameters: LogVolWithSkew,
    ) -> LatentVol:
        dt = condition.dt
        eps_log_vol = jrandom.normal(key)

        log_vol_mean = particle.current_log_vol + dt * (
            parameters.mean_reversion
            * (parameters.long_term_log_vol - particle.current_log_vol)
        )

        noise = parameters.std_log_vol * jnp.sqrt(dt) * eps_log_vol
        next_log_vol = log_vol_mean + noise

        return LatentVol(
            current_log_vol=next_log_vol,
            last_log_vol=particle.current_log_vol,
        )

    @staticmethod
    def log_p(
        particle: LatentVol,
        next_particle: LatentVol,
        condition: LastUnderlying,
        parameters: LogVolWithSkew,
    ) -> Scalar:
        dt = condition.dt

        log_vol_mean = particle.current_log_vol + dt * (
            parameters.mean_reversion
            * (parameters.long_term_log_vol - particle.current_log_vol)
        )

        # next_particle.last_log_vol == particle.current_log_vol
        return jstats.norm.logpdf(
            next_particle.current_log_vol,
            loc=log_vol_mean,
            scale=parameters.std_log_vol * jnp.sqrt(dt),
        )


class LogReturn(Emission[LatentVol, LastUnderlying, Underlying, LogVolWithSkew]):
    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: LatentVol,
        condition: LastUnderlying,
        parameters: LogVolRW,
    ) -> Underlying:
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(particle.last_log_vol)
        log_return = jrandom.normal(key) * return_scale
        return Underlying(underlying=condition.last_underlying * jnp.exp(log_return))

    @staticmethod
    def log_p(
        particle: LatentVol,
        observation: Underlying,
        condition: LastUnderlying,
        parameters: LogVolRW,
    ):
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(particle.last_log_vol)
        log_return = jnp.log(observation.underlying) - jnp.log(
            condition.last_underlying
        )
        return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)


class SkewLogReturn(Emission[LatentVol, LastUnderlying, Underlying, LogVolWithSkew]):
    @staticmethod
    def return_mean_and_scale(
        particle: LatentVol,
        condition: LastUnderlying,
        parameters: LogVolWithSkew,
    ) -> tuple[Scalar, Scalar]:
        dt = condition.dt

        last_vol = jnp.exp(particle.last_log_vol)
        last_var = jnp.exp(2 * particle.last_log_vol)

        log_vol_mean = particle.last_log_vol + dt * (
            parameters.mean_reversion
            * (parameters.long_term_log_vol - particle.last_log_vol)
        )

        return_mean = -0.5 * dt * last_var
        return_mean -= (
            parameters.skew
            * (last_vol / parameters.std_log_vol)
            * (particle.current_log_vol - log_vol_mean)
        )

        return_scale = jnp.sqrt(condition.dt) * last_vol * (1 - parameters.skew**2)

        return return_mean, return_scale

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: LatentVol,
        condition: LastUnderlying,
        parameters: LogVolWithSkew,
    ) -> Underlying:

        return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
            particle, condition, parameters
        )

        log_return = jrandom.normal(key) * return_scale + return_mean

        return Underlying(underlying=condition.last_underlying * jnp.exp(log_return))

    @staticmethod
    def log_p(
        particle: LatentVol,
        observation: Underlying,
        condition: LastUnderlying,
        parameters: LogVolWithSkew,
    ) -> Scalar:
        return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
            particle, condition, parameters
        )

        log_return = jnp.log(observation.underlying) - jnp.log(
            condition.last_underlying
        )

        return jstats.norm.logpdf(log_return, loc=return_mean, scale=return_scale)


class SimpleStochasticVol(Target[LatentVol, LastUnderlying, Underlying, LogVolRW]):
    prior = UniformStart
    transition = RandomWalk
    emission = LogReturn

    @staticmethod
    def emission_to_condition(
        observation: Underlying,
        condition: LastUnderlying,
        parameters: LogVolRW,
    ):
        return LastUnderlying(
            dt=condition.dt,
            last_underlying=observation.underlying,
        )

    @staticmethod
    def reference_emission(parameters: LogVolRW):
        return Underlying(underlying=parameters.initial_underlying)


class SkewStochasticVol(Target[LatentVol, LastUnderlying, Underlying, LogVolWithSkew]):
    prior = UniformStart
    transition = SkewRandomWalk
    emission = SkewLogReturn

    @staticmethod
    def emission_to_condition(
        observation: Underlying,
        condition: LastUnderlying,
        parameters: LogVolRW,
    ):
        return LastUnderlying(
            dt=condition.dt,
            last_underlying=observation.underlying,
        )

    @staticmethod
    def reference_emission(parameters: LogVolWithSkew):
        return Underlying(underlying=parameters.initial_underlying)
