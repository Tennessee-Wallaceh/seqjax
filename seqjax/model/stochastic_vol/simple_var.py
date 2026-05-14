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
    validate_sequential_model,
    ConditionContext,
    LatentContext,
    ObservationContext,
    ParameterizationProtocol,
    SequentialModelProtocol,
)
from seqjax.model.typing import Parameters, NoCondition, NoHyper

from .types import LatentVar, LogReturnObs, LogVarParams


prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0

# latent state is an annualized variance
# observations are in annualized terms
# but sampling interval may be varying, as such ar + std_log_var parameters 
# adjust for timescale -> minute sampling will be less varying + higher ar
# than daily
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
class SimpleStochasticVar(
    SequentialModelProtocol[
        LatentVar,
        LogReturnObs,
        NoCondition,
        LogVarParams,
    ]
):
    prior_order: int = prior_order
    transition_order: int = transition_order
    emission_order: int = emission_order
    observation_dependency: int = observation_dependency

    latent_cls: type[LatentVar] = latent_cls
    observation_cls: type[LogReturnObs] = observation_cls
    parameter_cls: type[LogVarParams] = parameter_cls
    condition_cls: type[NoCondition] = condition_cls

    latent_context: typing.Callable[..., LatentContext[LatentVar]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[LogReturnObs]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[NoCondition]] = condition_context

    prior_sample = staticmethod(prior_sample)
    prior_log_prob = staticmethod(prior_log_prob)
    transition_sample = staticmethod(transition_sample)
    transition_log_prob = staticmethod(transition_log_prob)
    emission_sample = staticmethod(emission_sample)
    emission_log_prob = staticmethod(emission_log_prob)


simple_stochastic_var_model = validate_sequential_model(SimpleStochasticVar())

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
@dataclass(frozen=True)
class LogVarPriorHyper:
    # defaults correspond to a uniform
    ar_mean: Scalar = field(default_factory=lambda: jnp.array(0.))
    ar_std: Scalar = field(default_factory=lambda: jnp.sqrt(jnp.array(1 / 3)))

    # note that the prior is a log-normal
    long_term_vol_mean: Scalar = field(default_factory=lambda: jnp.array(0.16))
    long_term_vol_std: Scalar = field(default_factory=lambda: jnp.array(0.3))

    # again, a log-normal prior
    std_log_var_mean: Scalar = field(default_factory=lambda: jnp.array(0.2))
    std_log_var_std: Scalar = field(default_factory=lambda: jnp.array(0.1))

    @property
    def ar_beta_ab(self):
        # map mean/std to beta parameters
        m = 0.5 * (self.ar_mean + 1.0)
        var_z = (self.ar_std / 2.0) ** 2
        concentration = m * (1.0 - m) / var_z - 1.0
        return m * concentration, (1.0 - m) * concentration
    
    @property
    def long_term_log_var_mean_std(self):
        # map to mean/std for gaussian in log var space
        cv2 = jnp.square(self.long_term_vol_std / self.long_term_vol_mean)

        log_vol_sd = jnp.sqrt(jnp.log1p(cv2))
        log_vol_loc = jnp.log(self.long_term_vol_mean) - 0.5 * jnp.square(log_vol_sd)

        long_term_log_var_mean = 2.0 * log_vol_loc
        long_term_log_var_sd = 2.0 * log_vol_sd

        return long_term_log_var_mean, long_term_log_var_sd
    
    @property
    def std_log_var_log_mean_std(self):
        # std_log_var ~ LogNormal(log_loc, log_sd)
        cv2 = jnp.square(self.std_log_var_std / self.std_log_var_mean)

        log_sd = jnp.sqrt(jnp.log1p(cv2))
        log_loc = (
            jnp.log(self.std_log_var_mean)
            - 0.5 * jnp.square(log_sd)
        )

        return log_loc, log_sd
    
@jax.tree_util.register_dataclass
@dataclass
class FullVarParameterization(
    ParameterizationProtocol[
        LogVarParams,
        UncLogVarParams,
        LogVarPriorHyper,
    ]
):
    """
    
    """
    _hyperparameters: LogVarPriorHyper = field(default_factory=LogVarPriorHyper)
    inference_parameter_cls: typing.ClassVar[type[UncLogVarParams]] = UncLogVarParams

    @property
    def hyperparameters(self):
        return jax.lax.stop_gradient(self._hyperparameters)
    
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
        k1, k2, k3 = jrandom.split(key, 3)

        ar_beta_a, ar_beta_b = self.hyperparameters.ar_beta_ab
        long_term_log_var_mean, long_term_log_var_std = self.hyperparameters.long_term_log_var_mean_std
        std_log_var_mean, std_log_var_std = self.hyperparameters.std_log_var_log_mean_std

        return self.from_model_parameters(LogVarParams(
            std_log_var=jnp.exp(std_log_var_mean + std_log_var_std * jrandom.normal(k1)),
            ar=2 * jrandom.beta(k2, ar_beta_a, ar_beta_b) - 1.0,
            long_term_log_var=long_term_log_var_mean + long_term_log_var_std * jrandom.normal(k3),
        ))
    
    def log_prob(self, inference_parameters: UncLogVarParams) -> Scalar:
        
        model_params = self.to_model_parameters(inference_parameters)

        # std log var
        std_log_var_mean, std_log_var_std = self.hyperparameters.std_log_var_log_mean_std
        std_lp = (
            jstats.norm.logpdf(
                jnp.log(model_params.std_log_var), 
                loc=std_log_var_mean, 
                scale=std_log_var_std
            )
            - jnp.log(model_params.std_log_var)
        )
        lad_std_log_var = jax.nn.log_sigmoid(inference_parameters.sft_inv_std_log_var)

        # long lerm log var
        long_term_log_var_mean, long_term_log_var_std = self.hyperparameters.long_term_log_var_mean_std
        long_term_lp = jstats.norm.logpdf(
            model_params.long_term_log_var,
            loc=long_term_log_var_mean,
            scale=long_term_log_var_std,
        )

        # ar
        ar01 = 0.5 * (model_params.ar + 1.0)
        ar_beta_a, ar_beta_b = self.hyperparameters.ar_beta_ab
        ar_lp = (
            jstats.beta.logpdf(ar01, a=ar_beta_a, b=ar_beta_b)
            - jnp.log(2.0)
        )
        lad_ar = jnp.log1p(-jnp.square(model_params.ar))

        return (
            std_lp
            + ar_lp
            + long_term_lp
            + lad_ar
            + lad_std_log_var
        )

@jax.tree_util.register_dataclass
@dataclass
class StochasticVarBayesian:
    target: typing.ClassVar = simple_stochastic_var_model
    parameterization : FullVarParameterization


def svar_full(hyperparameters: LogVarPriorHyper = LogVarPriorHyper()) -> StochasticVarBayesian:
    return StochasticVarBayesian(
        parameterization=FullVarParameterization(hyperparameters)
    )
