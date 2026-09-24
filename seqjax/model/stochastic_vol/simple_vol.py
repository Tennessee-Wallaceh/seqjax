from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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

from .common import StdLogVolPrior, StochVolParamPrior, lv_to_std_only, random_walk_loc_scale
from .types import LVolStd, LatentVol, LogReturnObs, LogVolRW, TimeIncrement


transition_latent_order = 1
emission_latent_order = 1
transition_observation_order = 0
emission_observation_order = 0

latent_cls = LatentVol
observation_cls = LogReturnObs
parameter_cls = LogVolRW
condition_cls = TimeIncrement

def prior_sample(
    key: PRNGKeyArray,
    parameters: LogVolRW,
) -> LatentContext[LatentVol]:
    _ = parameters
    mu = jnp.log(jnp.array(0.1))
    sigma = jnp.array(1.6) / jnp.sqrt(2.0 * 6.0)
    start_lv = LatentVol(log_vol=(mu + sigma * jrandom.normal(key)))
    return LatentContext.from_values(start_lv, length=max(transition_latent_order, emission_latent_order))


def prior_log_prob(
    latent: LatentContext[LatentVol],
    parameters: LogVolRW,
) -> Scalar:
    _ = parameters
    mu = jnp.log(jnp.array(0.1))
    sigma = jnp.array(1.6) / jnp.sqrt(2.0 * 6.0)
    return jstats.norm.logpdf(latent[-1].log_vol, loc=mu, scale=sigma)


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVol],
    parameters: LogVolRW,
    condition: TimeIncrement,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeIncrement],
) -> LatentVol:
    _ = observation_history
    loc, scale = random_walk_loc_scale(latent_history[-1], condition, parameters)
    return LatentVol(log_vol=loc + scale * jrandom.normal(key))


def transition_log_prob(
    latent_history: LatentContext[LatentVol],
    latent: LatentVol,
    parameters: LogVolRW,
    condition: TimeIncrement,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeIncrement],
) -> Scalar:
    _ = observation_history
    loc, scale = random_walk_loc_scale(latent_history[-1], condition, parameters)
    return jstats.norm.logpdf(latent.log_vol, loc=loc, scale=scale)


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVol],
    parameters: LogVolRW,
    condition: TimeIncrement,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeIncrement],
) -> LogReturnObs:
    _ = observation_history
    _ = parameters
    current_latent = latent_history[-1]
    return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_latent.log_vol)
    log_return = jrandom.normal(key) * return_scale
    return LogReturnObs(log_return=log_return)


def emission_log_prob(
    latent_history: LatentContext[LatentVol],
    observation: LogReturnObs,
    parameters: LogVolRW,
    condition: TimeIncrement,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeIncrement],
) -> Scalar:
    _ = observation_history
    _ = parameters
    current_latent = latent_history[-1]
    return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_latent.log_vol)
    return jstats.norm.logpdf(observation.log_return, loc=0.0, scale=return_scale)


simple_stochastic_vol_model = SequentialModel(
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
class SimpleStochasticVolBayesian:
    inference_parameter_cls: typing.ClassVar[type[LogVolRW]] = LogVolRW
    target: typing.ClassVar = simple_stochastic_vol_model
    parameter_prior: typing.ClassVar = StochVolParamPrior()
    convert_to_model_parameters = staticmethod(lambda parameters: parameters)


@dataclass
class SimpleStochasticVolBayesianStdLogVol:
    ref_params: LogVolRW
    inference_parameter_cls: typing.ClassVar[type[LVolStd]] = LVolStd
    target: typing.ClassVar = simple_stochastic_vol_model
    parameter_prior: typing.ClassVar = StdLogVolPrior()

    def __post_init__(self):
        self.convert_to_model_parameters = staticmethod(
            partial(lv_to_std_only, ref_params=self.ref_params)
        )


def make_constant_time_increments(
    sequence_length: int,
    *,
    dt: float = 1.0,
) -> TimeIncrement:
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    dt_value = jnp.asarray(dt, dtype=jnp.float32)
    increments = jnp.full((sequence_length,), dt_value, dtype=dt_value.dtype)
    return TimeIncrement(dt=increments)
