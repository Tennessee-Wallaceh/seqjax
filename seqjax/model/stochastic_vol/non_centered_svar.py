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
from seqjax.model.typing import Latent, Parameters, NoCondition, NoHyper

from .common import StochVarARPrior, StochVarFullPrior, StochVarPrior, lvar_from_ar_only, lvar_from_std_only
from .types import LatentVar, LogReturnObs, LogVarAR, LogVarParams, LogVarStd
from .simple_var import UncLogVarParams, FullVarParameterization

class NonCenteredLatentVar(Latent):
    z: Scalar
    _shape_template: typing.ClassVar = OrderedDict(
        z=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0

latent_cls = NonCenteredLatentVar
observation_cls = LogReturnObs
parameter_cls = LogVarParams
condition_cls = NoCondition

latent_context = partial(LatentContext, length=transition_order)
observation_context = partial(ObservationContext, length=observation_dependency)
condition_context = partial(ConditionContext, length=0)

def _stationary_scale_nc(parameters: LogVarParams) -> Scalar:
    return jnp.sqrt(1.0 / (1 - jnp.square(parameters.ar)))

def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[NoCondition],
    parameters: LogVarParams,
) -> LatentContext[NonCenteredLatentVar]:
    _ = conditions
    sigma = _stationary_scale_nc(parameters)
    start_z = NonCenteredLatentVar(z=sigma * jrandom.normal(key))
    return latent_context((start_z,))

def prior_log_prob(
    latent: LatentContext[NonCenteredLatentVar],
    conditions: ConditionContext[NoCondition],
    parameters: LogVarParams,
) -> Scalar:
    _ = conditions
    sigma = _stationary_scale_nc(parameters)
    return jstats.norm.logpdf(latent[0].z, loc=0.0, scale=sigma)


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[NonCenteredLatentVar],
    condition: NoCondition,
    parameters: LogVarParams,
) -> NonCenteredLatentVar:
    _ = condition
    last_z = latent_history[0].z
    loc = parameters.ar * last_z
    return NonCenteredLatentVar(z=loc + jrandom.normal(key))

def transition_log_prob(
    latent_history: LatentContext[NonCenteredLatentVar],
    latent: NonCenteredLatentVar,
    condition: NoCondition,
    parameters: LogVarParams,
) -> Scalar:
    _ = condition
    last_z = latent_history[0].z
    loc = parameters.ar * last_z
    return jstats.norm.logpdf(latent.z, loc=loc, scale=1.0)

def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[NonCenteredLatentVar],
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: LogVarParams,
) -> LogReturnObs:
    _ = observation_history
    _ = condition
    current_z = latent_history[0].z
    current_log_var = (
        parameters.long_term_log_var
        + parameters.std_log_var * current_z
    )
    return_scale = jnp.exp(0.5 * current_log_var)
    return LogReturnObs(log_return=jrandom.normal(key) * return_scale)

def emission_log_prob(
    latent_history: LatentContext[NonCenteredLatentVar],
    observation: LogReturnObs,
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: LogVarParams,
) -> Scalar:
    _ = observation_history
    _ = condition
    current_z = latent_history[0].z
    current_log_var = (
        parameters.long_term_log_var
        + parameters.std_log_var * current_z
    )
    return_scale = jnp.exp(0.5 * current_log_var)
    return jstats.norm.logpdf(
        observation.log_return,
        loc=0.0,
        scale=return_scale,
    )

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NCStochasticVar:
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

@jax.tree_util.register_dataclass
@dataclass
class NCStochasticVarBayesian:
    target: typing.ClassVar = validate_sequential_model(NCStochasticVar())
    parameterization : FullVarParameterization


def svar_nc_full(hyperparameters: typing.Any = NoHyper()) -> NCStochasticVarBayesian:
    return NCStochasticVarBayesian(
        parameterization=FullVarParameterization()
    )
