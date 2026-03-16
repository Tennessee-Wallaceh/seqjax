from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import (
    ConditionContext,
    LatentContext,
    ObservationContext,
    SequentialModelProtocol,
    validate_sequential_model,
)
from seqjax.model.typing import NoCondition

from .common import StdLogVolPrior, StochVolParamPrior, lv_to_std_only, random_walk_loc_scale
from .types import LVolStd, LatentVol, LogReturnObs, LogVolRW, TimeIncrement


prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0

latent_cls = LatentVol
observation_cls = LogReturnObs
parameter_cls = LogVolRW
condition_cls = TimeIncrement

latent_context: typing.Callable[[tuple[LatentVol]], LatentContext[LatentVol]]
latent_context = partial(LatentContext, length=transition_order)
observation_context: typing.Callable[[tuple], ObservationContext[LogReturnObs]]
observation_context = partial(ObservationContext, length=observation_dependency)
condition_context: typing.Callable[[tuple[TimeIncrement]], ConditionContext[TimeIncrement]]
condition_context = partial(ConditionContext, length=prior_order)


def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[TimeIncrement],
    parameters: LogVolRW,
) -> LatentContext[LatentVol]:
    _ = conditions
    _ = parameters
    mu = jnp.log(jnp.array(0.1))
    sigma = jnp.array(1.6) / jnp.sqrt(2.0 * 6.0)
    start_lv = LatentVol(log_vol=(mu + sigma * jrandom.normal(key)))
    return latent_context((start_lv,))


def prior_log_prob(
    latent: LatentContext[LatentVol],
    conditions: ConditionContext[TimeIncrement],
    parameters: LogVolRW,
) -> Scalar:
    _ = conditions
    _ = parameters
    mu = jnp.log(jnp.array(0.1))
    sigma = jnp.array(1.6) / jnp.sqrt(2.0 * 6.0)
    return jstats.norm.logpdf(latent[0].log_vol, loc=mu, scale=sigma)


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVol],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> LatentVol:
    loc, scale = random_walk_loc_scale(latent_history[0], condition, parameters)
    return LatentVol(log_vol=loc + scale * jrandom.normal(key))


def transition_log_prob(
    latent_history: LatentContext[LatentVol],
    latent: LatentVol,
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> Scalar:
    loc, scale = random_walk_loc_scale(latent_history[0], condition, parameters)
    return jstats.norm.logpdf(latent.log_vol, loc=loc, scale=scale)


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVol],
    observation_history: ObservationContext[LogReturnObs],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> LogReturnObs:
    _ = observation_history
    _ = parameters
    current_latent = latent_history[0]
    return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_latent.log_vol)
    log_return = jrandom.normal(key) * return_scale
    return LogReturnObs(log_return=log_return)


def emission_log_prob(
    latent_history: LatentContext[LatentVol],
    observation: LogReturnObs,
    observation_history: ObservationContext[LogReturnObs],
    condition: TimeIncrement,
    parameters: LogVolRW,
) -> Scalar:
    _ = observation_history
    _ = parameters
    current_latent = latent_history[0]
    return_scale = jnp.sqrt(condition.dt) * jnp.exp(current_latent.log_vol)
    return jstats.norm.logpdf(observation.log_return, loc=0.0, scale=return_scale)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SimpleStochasticVol(
    SequentialModelProtocol[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolRW,
    ]
):
    prior_order: int = prior_order
    transition_order: int = transition_order
    emission_order: int = emission_order
    observation_dependency: int = observation_dependency

    latent_cls: type[LatentVol] = latent_cls
    observation_cls: type[LogReturnObs] = observation_cls
    parameter_cls: type[LogVolRW] = parameter_cls
    condition_cls: type[TimeIncrement] = condition_cls

    latent_context: typing.Callable[..., LatentContext[LatentVol]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[LogReturnObs]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[TimeIncrement]] = condition_context

    prior_sample = staticmethod(prior_sample)
    prior_log_prob = staticmethod(prior_log_prob)
    transition_sample = staticmethod(transition_sample)
    transition_log_prob = staticmethod(transition_log_prob)
    emission_sample = staticmethod(emission_sample)
    emission_log_prob = staticmethod(emission_log_prob)


simple_stochastic_vol_model = validate_sequential_model(SimpleStochasticVol())


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
    prior_order: int | None = None,
) -> TimeIncrement:
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    effective_prior_order = (
        simple_stochastic_vol_model.prior_order if prior_order is None else prior_order
    )
    if effective_prior_order < 1:
        raise ValueError(
            f"prior_order must be >= 1, got {effective_prior_order}",
        )

    required_length = sequence_length + effective_prior_order - 1
    dt_value = jnp.asarray(dt, dtype=jnp.float32)
    increments = jnp.full((required_length,), dt_value, dtype=dt_value.dtype)
    return TimeIncrement(dt=increments)
