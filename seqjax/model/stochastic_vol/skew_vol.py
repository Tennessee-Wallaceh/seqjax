from __future__ import annotations

from dataclasses import dataclass
import typing

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import (
    ObservedHistoryContext,
    LatentContext,
    SequentialModel,
)

from .common import SkewStochVolParamPrior, random_walk_loc_scale, skew_return_mean_and_scale
from .types import LatentVol, LogReturnObs, LogVolWithSkew, TimeIncrement


transition_latent_order = 1
emission_latent_order = 1
transition_observation_order = 0
emission_observation_order = 0

latent_cls = LatentVol
observation_cls = LogReturnObs
parameter_cls = LogVolWithSkew
condition_cls = TimeIncrement

def prior_sample(
    key: PRNGKeyArray,
    parameters: LogVolWithSkew,
) -> LatentContext[LatentVol, typing.Literal[1]]:
    mu = jnp.array(-2.0)
    sigma = jnp.array(0.5)

    start_lv = LatentVol(log_vol=mu + sigma * jrandom.normal(key))
    return LatentContext.from_values(start_lv, length=1)


def prior_log_prob(
    latent: LatentContext[LatentVol, typing.Literal[1]],
    parameters: LogVolWithSkew,
) -> Scalar:
    mu = jnp.array(-2.0)
    sigma = jnp.array(0.5)

    return sum(
        jstats.norm.logpdf(value.log_vol, loc=mu, scale=sigma)
        for value in latent.to_tuple()
    )


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVol, typing.Literal[1]],
    parameters: LogVolWithSkew,
    condition: TimeIncrement,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeIncrement, typing.Literal[0]],
) -> LatentVol:
    _ = observation_history
    loc, scale = random_walk_loc_scale(latent_history[-1], condition, parameters)
    return LatentVol(log_vol=loc + scale * jrandom.normal(key))


def transition_log_prob(
    latent: LatentVol,
    latent_history: LatentContext[LatentVol, typing.Literal[1]],
    parameters: LogVolWithSkew,
    condition: TimeIncrement,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeIncrement, typing.Literal[0]],
) -> Scalar:
    _ = observation_history
    loc, scale = random_walk_loc_scale(latent_history[-1], condition, parameters)
    return jstats.norm.logpdf(latent.log_vol, loc=loc, scale=scale)


def emission_sample(
    key: PRNGKeyArray,
    current_latent: LatentVol,
    parameters: LogVolWithSkew,
    condition: TimeIncrement,
    latent_history: LatentContext[LatentVol, typing.Literal[1]],
    observation_history: ObservedHistoryContext[LogReturnObs, TimeIncrement, typing.Literal[0]],
) -> LogReturnObs:
    _ = observation_history
    return_mean, return_scale = skew_return_mean_and_scale(
        latent_history[-1],
        current_latent,
        condition,
        parameters,
    )
    log_return = jrandom.normal(key) * return_scale + return_mean
    return LogReturnObs(log_return=log_return)


def emission_log_prob(
    observation: LogReturnObs,
    current_latent: LatentVol,
    parameters: LogVolWithSkew,
    condition: TimeIncrement,
    latent_history: LatentContext[LatentVol, typing.Literal[1]],
    observation_history: ObservedHistoryContext[LogReturnObs, TimeIncrement, typing.Literal[0]],
) -> Scalar:
    _ = observation_history
    return_mean, return_scale = skew_return_mean_and_scale(
        latent_history[-1],
        current_latent,
        condition,
        parameters,
    )
    return jstats.norm.logpdf(observation.log_return, loc=return_mean, scale=return_scale)


skew_stochastic_vol_model = SequentialModel(
    latent_cls=latent_cls,
    observation_cls=observation_cls,
    parameter_cls=parameter_cls,
    condition_cls=condition_cls,
    transition_latent_order=transition_latent_order,
    transition_observation_order=transition_observation_order,
    emission_latent_order=emission_latent_order,
    emission_observation_order=emission_observation_order,
    prior_sample=prior_sample,
    prior_log_prob=prior_log_prob,
    transition_sample=transition_sample,
    transition_log_prob=transition_log_prob,
    emission_sample=emission_sample,
    emission_log_prob=emission_log_prob,
)


@dataclass
class SkewStochasticVolBayesian:
    inference_parameter_cls: typing.ClassVar[type[LogVolWithSkew]] = LogVolWithSkew
    target: typing.ClassVar = skew_stochastic_vol_model
    parameter_prior: typing.ClassVar = SkewStochVolParamPrior()
    convert_to_model_parameters = staticmethod(lambda parameters: parameters)
