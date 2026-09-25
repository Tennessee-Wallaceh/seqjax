from collections import OrderedDict
from dataclasses import dataclass
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import (
    ObservedHistoryContext,
    LatentContext,
    SequentialModel,
)
from seqjax.model.typing import Latent, NoCondition, NoHyper

from .types import LogReturnObs, LogVarParams
from .simple_var import FullVarParameterization

class NonCenteredLatentVar(Latent):
    z: Scalar
    _shape_template: typing.ClassVar = OrderedDict(
        z=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

transition_latent_order = 1
emission_latent_order = 0
transition_observation_order = 0
emission_observation_order = 0

latent_cls = NonCenteredLatentVar
observation_cls = LogReturnObs
parameter_cls = LogVarParams
condition_cls = NoCondition

def _stationary_scale_nc(parameters: LogVarParams) -> Scalar:
    return jnp.sqrt(1.0 / (1 - jnp.square(parameters.ar)))

def prior_sample(
    key: PRNGKeyArray,
    parameters: LogVarParams,
) -> LatentContext[NonCenteredLatentVar, typing.Literal[1]]:
    sigma = _stationary_scale_nc(parameters)
    start_z = NonCenteredLatentVar(z=sigma * jrandom.normal(key))
    return LatentContext.from_values(start_z, length=1)

def prior_log_prob(
    latent: LatentContext[NonCenteredLatentVar, typing.Literal[1]],
    parameters: LogVarParams,
) -> Scalar:
    sigma = _stationary_scale_nc(parameters)
    return jstats.norm.logpdf(latent[-1].z, loc=0.0, scale=sigma)


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[NonCenteredLatentVar, typing.Literal[1]],
    parameters: LogVarParams,
    condition: NoCondition,
    observation_history: ObservedHistoryContext[LogReturnObs, NoCondition, typing.Literal[0]],
) -> NonCenteredLatentVar:
    _ = observation_history
    _ = condition
    last_z = latent_history[-1].z
    loc = parameters.ar * last_z
    return NonCenteredLatentVar(z=loc + jrandom.normal(key))

def transition_log_prob(
    latent: NonCenteredLatentVar,
    latent_history: LatentContext[NonCenteredLatentVar, typing.Literal[1]],
    parameters: LogVarParams,
    condition: NoCondition,
    observation_history: ObservedHistoryContext[LogReturnObs, NoCondition, typing.Literal[0]],
) -> Scalar:
    _ = observation_history
    _ = condition
    last_z = latent_history[-1].z
    loc = parameters.ar * last_z
    return jstats.norm.logpdf(latent.z, loc=loc, scale=1.0)

def emission_sample(
    key: PRNGKeyArray,
    current_latent: NonCenteredLatentVar,
    parameters: LogVarParams,
    condition: NoCondition,
    latent_history: LatentContext[NonCenteredLatentVar, typing.Literal[1]],
    observation_history: ObservedHistoryContext[LogReturnObs, NoCondition, typing.Literal[0]],
) -> LogReturnObs:
    _ = observation_history
    _ = condition
    current_log_var = (
        parameters.long_term_log_var
        + parameters.std_log_var * current_latent.z
    )
    return_scale = jnp.exp(0.5 * current_log_var)
    return LogReturnObs(log_return=jrandom.normal(key) * return_scale)

def emission_log_prob(
    observation: LogReturnObs,
    current_latent: NonCenteredLatentVar,
    parameters: LogVarParams,
    condition: NoCondition,
    latent_history: LatentContext[NonCenteredLatentVar, typing.Literal[1]],
    observation_history: ObservedHistoryContext[LogReturnObs, NoCondition, typing.Literal[0]],
) -> Scalar:
    _ = observation_history
    _ = condition
    current_log_var = (
        parameters.long_term_log_var
        + parameters.std_log_var * current_latent.z
    )
    return_scale = jnp.exp(0.5 * current_log_var)
    return jstats.norm.logpdf(
        observation.log_return,
        loc=0.0,
        scale=return_scale,
    )

nc_stochastic_var_model = SequentialModel(
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

@jax.tree_util.register_dataclass
@dataclass
class NCStochasticVarBayesian:
    target: typing.ClassVar = nc_stochastic_var_model
    parameterization : FullVarParameterization


def svar_nc_full(hyperparameters: typing.Any = NoHyper()) -> NCStochasticVarBayesian:
    return NCStochasticVarBayesian(
        parameterization=FullVarParameterization()
    )
