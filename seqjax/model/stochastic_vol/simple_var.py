from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import typing

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import ConditionContext, LatentContext, ObservationContext
from seqjax.model.typing import NoCondition

from .common import StochVarARPrior, StochVarFullPrior, StochVarPrior, lvar_from_ar_only, lvar_from_std_only
from .types import LatentVar, LogReturnObs, LogVarAR, LogVarParams, LogVarStd


prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0

latent_cls = LatentVar
observation_cls = LogReturnObs
parameter_cls = LogVarParams
condition_cls = NoCondition

latent_context = partial(LatentContext, length=transition_order)
observation_context = partial(ObservationContext, length=observation_dependency)
condition_context = partial(ConditionContext, length=0)


def _stationary_scale(parameters: LogVarParams) -> Scalar:
    return jnp.sqrt(jnp.square(parameters.std_log_var) / (1 - jnp.square(parameters.ar)))


def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[NoCondition],
    parameters: LogVarParams,
) -> LatentContext[LatentVar]:
    _ = conditions
    sigma = _stationary_scale(parameters)
    mu = parameters.long_term_log_var
    start_lv = LatentVar(log_var=mu + sigma * jrandom.normal(key))
    return latent_context((start_lv,))


def prior_log_prob(
    latent: LatentContext[LatentVar],
    conditions: ConditionContext[NoCondition],
    parameters: LogVarParams,
) -> Scalar:
    _ = conditions
    sigma = _stationary_scale(parameters)
    mu = parameters.long_term_log_var
    return jstats.norm.logpdf(latent[0].log_var, loc=mu, scale=sigma)


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVar],
    condition: NoCondition,
    parameters: LogVarParams,
) -> LatentVar:
    _ = condition
    last_log_var = latent_history[0]
    loc = parameters.long_term_log_var + parameters.ar * (
        last_log_var.log_var - parameters.long_term_log_var
    )
    return LatentVar(log_var=loc + parameters.std_log_var * jrandom.normal(key))


def transition_log_prob(
    latent_history: LatentContext[LatentVar],
    latent: LatentVar,
    condition: NoCondition,
    parameters: LogVarParams,
) -> Scalar:
    _ = condition
    last_log_var = latent_history[0]
    loc = parameters.long_term_log_var + parameters.ar * (
        last_log_var.log_var - parameters.long_term_log_var
    )
    return jstats.norm.logpdf(latent.log_var, loc=loc, scale=parameters.std_log_var)


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVar],
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: LogVarParams,
) -> LogReturnObs:
    _ = observation_history
    _ = condition
    _ = parameters
    current_latent = latent_history[0]
    return_scale = jnp.exp(0.5 * current_latent.log_var)
    return LogReturnObs(log_return=jrandom.normal(key) * return_scale)


def emission_log_prob(
    latent_history: LatentContext[LatentVar],
    observation: LogReturnObs,
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: LogVarParams,
) -> Scalar:
    _ = observation_history
    _ = condition
    _ = parameters
    current_latent = latent_history[0]
    return_scale = jnp.exp(0.5 * current_latent.log_var)
    return jstats.norm.logpdf(observation.log_return, loc=0.0, scale=return_scale)


class SimpleStochasticVar:
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
class StochasticVarBayesian:
    inference_parameter_cls: typing.ClassVar[type[LogVarParams]] = LogVarParams
    target: typing.ClassVar = SimpleStochasticVar()
    parameter_prior: typing.ClassVar = StochVarFullPrior()
    convert_to_model_parameters = staticmethod(lambda parameters: parameters)


@dataclass
class SimpleStochasticVarBayesian:
    ref_params: LogVarParams
    inference_parameter_cls: typing.ClassVar[type[LogVarStd]] = LogVarStd
    target: typing.ClassVar = SimpleStochasticVar()
    parameter_prior: typing.ClassVar = StochVarPrior()

    def __post_init__(self):
        self.convert_to_model_parameters = staticmethod(
            partial(lvar_from_std_only, ref_params=self.ref_params)
        )


@dataclass
class ARStochasticVarBayesian:
    ref_params: LogVarParams
    inference_parameter_cls: typing.ClassVar[type[LogVarAR]] = LogVarAR
    target: typing.ClassVar = SimpleStochasticVar()
    parameter_prior: typing.ClassVar = StochVarARPrior()

    def __post_init__(self):
        self.convert_to_model_parameters = staticmethod(
            partial(lvar_from_ar_only, ref_params=self.ref_params)
        )
