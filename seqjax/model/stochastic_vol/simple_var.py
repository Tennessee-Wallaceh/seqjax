from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import (
    validate_sequential_model, ConditionContext, LatentContext, ObservationContext, ParameterizationProtocol
)
from seqjax.model.typing import Parameters, NoCondition, NoHyper

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

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
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

class UncLogVarParams(Parameters):
    sft_inv_std_log_var: Scalar
    logit_ar: Scalar
    long_term_log_var: Scalar
    _shape_template: typing.ClassVar = OrderedDict(
        logit_ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        sft_inv_std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

@jax.tree_util.register_dataclass
@dataclass
class FullVarParameterization(
    ParameterizationProtocol[
        LogVarParams,
        UncLogVarParams,
        NoHyper,
    ]
):
    inference_parameter_cls: type[UncLogVarParams] = UncLogVarParams
    hyperparameters: NoHyper = field(default_factory=NoHyper)

    def to_model_parameters(self, inference_parameters: UncLogVarParams) -> LogVarParams:
        return LogVarParams(
            std_log_var=jax.nn.softplus(inference_parameters.sft_inv_std_log_var),
            ar=jnp.tanh(inference_parameters.logit_ar),
            long_term_log_var=inference_parameters.long_term_log_var,
        )

    def from_model_parameters(self, model_parameters: LogVarParams) -> UncLogVarParams:
        return UncLogVarParams(
            sft_inv_std_log_var=jnp.log(jnp.expm1(model_parameters.std_log_var)),
            logit_ar=jnp.arctanh(model_parameters.ar),
            long_term_log_var=model_parameters.long_term_log_var,
        )

    def sample(self, key: PRNGKeyArray) -> UncLogVarParams:
        long_term_log_var_mean = 2 * jnp.log(jnp.array(0.16))
        k1, k2, k3 = jrandom.split(key, 3)
        return self.from_model_parameters(LogVarParams(
            std_log_var=10 / jrandom.gamma(k1, 10),
            ar=jrandom.uniform(k2, minval=-1, maxval=1),
            long_term_log_var=long_term_log_var_mean + jrandom.normal(k3),
        ))


    def log_prob(self, inference_parameters: UncLogVarParams) -> Scalar:
        long_term_log_var_mean = 2 * jnp.log(jnp.array(0.16))
        model_params = self.to_model_parameters(inference_parameters)
        lad_ar = jnp.log1p(-jnp.square(model_params.ar))
        lad_std_log_var = jax.nn.log_sigmoid(inference_parameters.sft_inv_std_log_var)
        return (
            jstats.gamma.logpdf(1 / model_params.std_log_var, 10, scale=1 / 10) 
            + lad_std_log_var
            + jstats.uniform.logpdf(model_params.ar, loc=-1.0, scale=2.0)
            + lad_ar
            + jstats.norm.logpdf(
                model_params.long_term_log_var,
                loc=long_term_log_var_mean
            )
        )

@jax.tree_util.register_dataclass
@dataclass
class StochasticVarBayesian:
    target: typing.ClassVar = validate_sequential_model(SimpleStochasticVar())
    parameterization : FullVarParameterization


def svar_full(hyperparameters: typing.Any = NoHyper()) -> StochasticVarBayesian:
    return StochasticVarBayesian(
        parameterization=FullVarParameterization()
    )
