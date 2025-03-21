from seqjax.target.base import Target, Scalar
from typing import NamedTuple, TypeVar, Protocol, Generic, Optional
from jaxtyping import Array, Float, PRNGKeyArray
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.random as jrandom
import jax


class LatentVol(NamedTuple):
    log_vol: Scalar


class LogVolRW(NamedTuple):
    log_vol_std: Scalar
    initial_underlying: Scalar


class Underlying(NamedTuple):
    underlying: Scalar


class LastUnderlying(NamedTuple):
    dt: Scalar  # time since last observation
    last_underlying: Scalar = jnp.array(jnp.nan)


def sample_prior(key: PRNGKeyArray, hyperparameters: LogVolRW) -> LatentVol:
    vol = jrandom.uniform(key, minval=0.01, maxval=0.3)  # in annualised terms
    return LatentVol(log_vol=jnp.log(vol))


def prior_log_p(particle: LatentVol, hyperparameters: LogVolRW) -> Float[Array, ""]:
    # re-parameterization
    base_log_p = jstats.uniform.logpdf(jnp.exp(particle.log_vol), loc=0.01, scale=0.29)
    log_jac_term = particle.log_vol
    return base_log_p + log_jac_term


def sample_transition(
    key: PRNGKeyArray,
    particle: LatentVol,
    condition: LastUnderlying,
    hyperparameters: LogVolRW,
) -> LatentVol:
    move_scale = jnp.sqrt(condition.dt) * hyperparameters.log_vol_std
    next_log_vol = particle.log_vol + jrandom.normal(key) * move_scale
    return LatentVol(log_vol=next_log_vol)


def transition_log_p(
    particle: LatentVol,
    next_particle: LatentVol,
    condition: LastUnderlying,
    hyperparameters: LogVolRW,
) -> Scalar:
    """
    log_vol[t2] ~ N(log_vol[t1], t * log_vol_std^2)
    """
    move_scale = jnp.sqrt(condition.dt) * hyperparameters.log_vol_std
    return jstats.norm.logpdf(
        next_particle.log_vol, loc=particle.log_vol, scale=move_scale
    )


def sample_emission(
    key: PRNGKeyArray,
    particle: LatentVol,
    condition: LastUnderlying,
    hyperparameters: LogVolRW,
) -> Underlying:
    return_scale = jnp.sqrt(condition.dt) * jnp.exp(particle.log_vol)
    log_return = jrandom.normal(key) * return_scale
    return Underlying(underlying=condition.last_underlying * jnp.exp(log_return))


def emission_log_p(
    particle: LatentVol,
    observation: Underlying,
    condition: LastUnderlying,
    hyperparameters: LogVolRW,
):
    return_scale = jnp.sqrt(condition.dt) * jnp.exp(particle.log_vol)
    log_return = jnp.log(observation.underlying) - jnp.log(condition.last_underlying)
    return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)


def emission_to_condition(
    observation: Underlying,
    condition: LastUnderlying,
    hyperparameters: LogVolRW,
):
    return LastUnderlying(
        dt=condition.dt,
        last_underlying=observation.underlying,
    )


def reference_emission(
    hyperparameters: LogVolRW,
):
    return Underlying(
        underlying=hyperparameters.initial_underlying,
    )


# Group together into a target
class SimpleStochasticVol(Target[LatentVol, Underlying, LastUnderlying, LogVolRW]):
    sample_prior = staticmethod(sample_prior)
    prior_log_p = staticmethod(prior_log_p)
    sample_transition = staticmethod(sample_transition)
    transition_log_p = staticmethod(transition_log_p)
    sample_emission = staticmethod(sample_emission)
    emission_log_p = staticmethod(emission_log_p)
    emission_to_condition = staticmethod(emission_to_condition)
    reference_emission = staticmethod(reference_emission)
