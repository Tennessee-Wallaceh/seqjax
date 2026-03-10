from __future__ import annotations

from dataclasses import dataclass
import typing
from functools import partial

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import ConditionContext, LatentContext, ObservationContext

from .common import SkewStochVolParamPrior, random_walk_loc_scale, skew_return_mean_and_scale
from .types import LatentVol, LogReturnObs, LogVolWithSkew, TimeIncrement


prior_order = 2
transition_order = 2
emission_order = 2
observation_dependency = 0

latent_cls = LatentVol
observation_cls = LogReturnObs
parameter_cls = LogVolWithSkew
condition_cls = TimeIncrement

latent_context = partial(LatentContext, length=transition_order)
observation_context = partial(ObservationContext, length=observation_dependency)
condition_context = partial(ConditionContext, length=prior_order)


def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[TimeIncrement],
    parameters: LogVolWithSkew,
) -> LatentContext[LatentVol]:
    mu = jnp.array(-2.0)
    sigma = jnp.array(0.5)

    start_key, trans_key = jrandom.split(key)
    start_lv = LatentVol(log_vol=mu + sigma * jrandom.normal(start_key))
    loc, scale = random_walk_loc_scale(start_lv, conditions[-1], parameters)
    next_lv = LatentVol(log_vol=loc + scale * jrandom.normal(trans_key))
    return latent_context((start_lv, next_lv))


def prior_log_prob(
    latent: LatentContext[LatentVol],
    conditions: ConditionContext[TimeIncrement],
    parameters: LogVolWithSkew,
) -> Scalar:
    mu = jnp.array(-2.0)
    sigma = jnp.array(0.5)

    base_log_p = jstats.norm.logpdf(latent[0].log_vol, loc=mu, scale=sigma)
    loc, scale = random_walk_loc_scale(latent[0], conditions[-1], parameters)
    rw_log_p = jstats.norm.logpdf(latent[-1].log_vol, loc=loc, scale=scale)
    return base_log_p + rw_log_p


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVol],
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> LatentVol:
    loc, scale = random_walk_loc_scale(latent_history[-1], condition, parameters)
    return LatentVol(log_vol=loc + scale * jrandom.normal(key))


def transition_log_prob(
    latent_history: LatentContext[LatentVol],
    latent: LatentVol,
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> Scalar:
    loc, scale = random_walk_loc_scale(latent_history[-1], condition, parameters)
    return jstats.norm.logpdf(latent.log_vol, loc=loc, scale=scale)


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVol],
    observation_history: ObservationContext[LogReturnObs],
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> LogReturnObs:
    _ = observation_history
    return_mean, return_scale = skew_return_mean_and_scale(
        latent_history[0],
        latent_history[-1],
        condition,
        parameters,
    )
    log_return = jrandom.normal(key) * return_scale + return_mean
    return LogReturnObs(log_return=log_return)


def emission_log_prob(
    latent_history: LatentContext[LatentVol],
    observation: LogReturnObs,
    observation_history: ObservationContext[LogReturnObs],
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> Scalar:
    _ = observation_history
    return_mean, return_scale = skew_return_mean_and_scale(
        latent_history[0],
        latent_history[-1],
        condition,
        parameters,
    )
    return jstats.norm.logpdf(observation.log_return, loc=return_mean, scale=return_scale)


class SkewStochasticVol:
    prior_order = prior_order
    transition_order = transition_order
    emission_order = emission_order
    observation_dependency = observation_dependency

    latent_cls = latent_cls
    observation_cls = observation_cls
    parameter_cls = parameter_cls
    condition_cls = condition_cls

    latent_context = staticmethod(latent_context)
    observation_context = staticmethod(observation_context)
    condition_context = staticmethod(condition_context)

    prior_sample = staticmethod(prior_sample)
    prior_log_prob = staticmethod(prior_log_prob)
    transition_sample = staticmethod(transition_sample)
    transition_log_prob = staticmethod(transition_log_prob)
    emission_sample = staticmethod(emission_sample)
    emission_log_prob = staticmethod(emission_log_prob)


@dataclass
class SkewStochasticVolBayesian:
    inference_parameter_cls: typing.ClassVar[type[LogVolWithSkew]] = LogVolWithSkew
    target: typing.ClassVar = SkewStochasticVol()
    parameter_prior: typing.ClassVar = SkewStochVolParamPrior()
    convert_to_model_parameters = staticmethod(lambda parameters: parameters)
