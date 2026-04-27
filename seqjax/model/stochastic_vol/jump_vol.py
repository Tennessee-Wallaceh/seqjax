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
from seqjax.model.typing import Parameters, NoCondition, NoHyper, HyperParameters

from .common import StochVarARPrior, StochVarFullPrior, StochVarPrior, lvar_from_ar_only, lvar_from_std_only
from .types import LatentVar, LogReturnObs, LogVarAR, LogVarParams, LogVarStd

prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0


class JumpVarParams(Parameters):
    std_log_var: Scalar
    ar: Scalar
    long_term_log_var: Scalar
    return_df: Scalar
    jump_prob: Scalar
    jump_mult: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        return_df=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        jump_prob=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        jump_mult=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

latent_cls = LatentVar
observation_cls = LogReturnObs
parameter_cls = JumpVarParams
condition_cls = NoCondition

latent_context = partial(LatentContext, length=transition_order)
observation_context = partial(ObservationContext, length=observation_dependency)
condition_context = partial(ConditionContext, length=0)

def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVar],
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: JumpVarParams,
) -> LogReturnObs:
    _ = observation_history
    _ = condition
    _ = parameters
    current_latent = latent_history[0]
    return_scale = jnp.exp(0.5 * current_latent.log_var)
    return LogReturnObs(log_return=jrandom.normal(key) * return_scale)

def _stationary_scale(parameters: JumpVarParams) -> Scalar:
    denom = jnp.maximum(1 - jnp.square(parameters.ar), 1e-4)
    return jnp.sqrt(jnp.square(parameters.std_log_var) / denom)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class JumpStochasticVar(
    SequentialModelProtocol[
        LatentVar,
        LogReturnObs,
        NoCondition,
        JumpVarParams,
    ]
):
    prior_order: int = prior_order
    transition_order: int = transition_order
    emission_order: int = emission_order
    observation_dependency: int = observation_dependency

    latent_cls: type[LatentVar] = latent_cls
    observation_cls: type[LogReturnObs] = observation_cls
    parameter_cls: type[JumpVarParams] = parameter_cls
    condition_cls: type[NoCondition] = condition_cls

    latent_context: typing.Callable[..., LatentContext[LatentVar]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[LogReturnObs]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[NoCondition]] = condition_context

    @staticmethod

    @staticmethod
    def prior_sample(
        key: PRNGKeyArray,
        conditions: ConditionContext[NoCondition],
        parameters: JumpVarParams,
    ) -> LatentContext[LatentVar]:
        _ = conditions
        sigma = _stationary_scale(parameters)
        mu = parameters.long_term_log_var
        start_lv = LatentVar(log_var=mu + sigma * jrandom.normal(key))
        return latent_context((start_lv,))

    @staticmethod
    def prior_log_prob(
        latent: LatentContext[LatentVar],
        conditions: ConditionContext[NoCondition],
        parameters: JumpVarParams,
    ) -> Scalar:
        _ = conditions
        sigma = _stationary_scale(parameters)
        mu = parameters.long_term_log_var
        return jstats.norm.logpdf(latent[0].log_var, loc=mu, scale=sigma)

    @staticmethod
    def transition_sample(
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentVar],
        condition: NoCondition,
        parameters: JumpVarParams,
    ) -> LatentVar:
        _ = condition
        last_log_var = latent_history[0]
        loc = parameters.long_term_log_var + parameters.ar * (
            last_log_var.log_var - parameters.long_term_log_var
        )
        return LatentVar(log_var=loc + parameters.std_log_var * jrandom.normal(key))

    @staticmethod
    def transition_log_prob(
        latent_history: LatentContext[LatentVar],
        latent: LatentVar,
        condition: NoCondition,
        parameters: JumpVarParams,
    ) -> Scalar:
        _ = condition
        last_log_var = latent_history[0]
        loc = parameters.long_term_log_var + parameters.ar * (
            last_log_var.log_var - parameters.long_term_log_var
        )
        return jstats.norm.logpdf(
            latent.log_var,
            loc=loc,
            scale=parameters.std_log_var,
        )
    
    @staticmethod
    def emission_sample(
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentVar],
        observation_history: ObservationContext[LogReturnObs],
        condition: NoCondition,
        parameters: JumpVarParams,
    ) -> LogReturnObs:
        _ = observation_history
        _ = condition

        current_latent = latent_history[0]

        base_scale = jnp.exp(0.5 * current_latent.log_var)
        std_t_scale = base_scale * jnp.sqrt(
            (parameters.return_df - 2.0) / parameters.return_df
        )
        jump_scale = base_scale * parameters.jump_mult

        k_branch, k_value = jrandom.split(key, 2)

        is_jump = jrandom.bernoulli(k_branch, p=parameters.jump_prob)

        t_draw = jrandom.t(k_value, df=parameters.return_df) * std_t_scale
        jump_draw = jrandom.normal(k_value) * jump_scale

        log_return = jnp.where(is_jump, jump_draw, t_draw)

        return LogReturnObs(log_return=log_return)

    @staticmethod
    def emission_log_prob(
        latent_history: LatentContext[LatentVar],
        observation: LogReturnObs,
        observation_history: ObservationContext[LogReturnObs],
        condition: NoCondition,
        parameters: JumpVarParams,
    ) -> Scalar:
        _ = observation_history
        _ = condition

        current_latent = latent_history[0]

        base_scale = jnp.exp(0.5 * current_latent.log_var)

        std_t_scale = base_scale * jnp.sqrt((parameters.return_df - 2.0) / parameters.return_df)

        log_no_jump = (
            jnp.log1p(-parameters.jump_prob)
            + jstats.t.logpdf(observation.log_return / std_t_scale, df=parameters.return_df)
            - jnp.log(std_t_scale)
        )

        jump_scale = base_scale * parameters.jump_mult

        log_jump = (
            jnp.log(parameters.jump_prob)
            + jstats.norm.logpdf(
                observation.log_return,
                loc=0.0,
                scale=jump_scale,
            )
        )

        return jax.scipy.special.logsumexp(jnp.array([log_no_jump, log_jump]))


class JumpVarHyper(HyperParameters):
    fixed_jump_prob: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        fixed_jump_prob=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class UncJumpVarParams(Parameters):
    sft_inv_stationary_scale: Scalar
    log_half_life_steps: Scalar
    long_term_log_var: Scalar
    sft_inv_return_df_m4: Scalar
    sft_inv_jump_mult_m2: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        sft_inv_stationary_scale=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        log_half_life_steps=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        sft_inv_return_df_m4=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        sft_inv_jump_mult_m2=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


def _softplus_inverse(y: Scalar) -> Scalar:
    return jnp.log(jnp.expm1(y))


@jax.tree_util.register_dataclass
@dataclass
class FullJumpVarParameterization(
    ParameterizationProtocol[
        JumpVarParams,
        UncJumpVarParams,
        JumpVarHyper,
    ]
):
    inference_parameter_cls: type[UncJumpVarParams] = UncJumpVarParams
    hyperparameters: JumpVarHyper = field(
        default_factory=lambda: JumpVarHyper(fixed_jump_prob=jnp.array(1e-3, dtype=jnp.float32))
    )

    def to_model_parameters(self, inference_parameters: UncJumpVarParams) -> JumpVarParams:
        stationary_scale = jax.nn.softplus(inference_parameters.sft_inv_stationary_scale)
        half_life_steps = jnp.exp(inference_parameters.log_half_life_steps)

        ar = jnp.exp(-jnp.log(2.0) / half_life_steps)
        std_log_var = stationary_scale * jnp.sqrt(jnp.maximum(1.0 - ar**2, 1e-8))

        return JumpVarParams(
            std_log_var=std_log_var,
            ar=ar,
            long_term_log_var=inference_parameters.long_term_log_var,
            return_df=4.0 + jax.nn.softplus(inference_parameters.sft_inv_return_df_m4),
            jump_prob=self.hyperparameters.fixed_jump_prob,
            jump_mult=2.0 + jax.nn.softplus(inference_parameters.sft_inv_jump_mult_m2),
        )

    def from_model_parameters(self, model_parameters: JumpVarParams) -> UncJumpVarParams:
        stationary_scale = model_parameters.std_log_var / jnp.sqrt(
            jnp.maximum(1.0 - model_parameters.ar**2, 1e-8)
        )
        half_life_steps = -jnp.log(2.0) / jnp.log(model_parameters.ar)

        return UncJumpVarParams(
            sft_inv_stationary_scale=_softplus_inverse(stationary_scale),
            log_half_life_steps=jnp.log(half_life_steps),
            long_term_log_var=model_parameters.long_term_log_var,
            sft_inv_return_df_m4=_softplus_inverse(model_parameters.return_df - 4.0),
            sft_inv_jump_mult_m2=_softplus_inverse(model_parameters.jump_mult - 2.0),
        )

    def sample(self, key: PRNGKeyArray) -> UncJumpVarParams:
        annual_vol_mean = 0.8
        long_term_log_var_mean = 2 * jnp.log(jnp.array(annual_vol_mean))

        k1, k2, k3, k4, k5 = jrandom.split(key, 5)

        stationary_scale = jnp.exp(jnp.log(0.5) + 0.35 * jrandom.normal(k1))
        half_life_steps = jnp.exp(jnp.log(100.0) + 0.75 * jrandom.normal(k2))
        long_term_log_var = long_term_log_var_mean + 0.5 * jrandom.normal(k3)
        return_df = 4.0 + jnp.exp(jnp.log(8.0) + 0.5 * jrandom.normal(k4))
        jump_mult = 2.0 + jnp.exp(jnp.log(1.0) + 0.5 * jrandom.normal(k5))

        ar = jnp.exp(-jnp.log(2.0) / half_life_steps)
        std_log_var = stationary_scale * jnp.sqrt(jnp.maximum(1.0 - jnp.square(ar), 1e-8))

        return self.from_model_parameters(
            JumpVarParams(
                std_log_var=std_log_var,
                ar=ar,
                long_term_log_var=long_term_log_var,
                return_df=return_df,
                jump_prob=self.hyperparameters.fixed_jump_prob,
                jump_mult=jump_mult,
            )
        )

    def log_prob(self, inference_parameters: UncJumpVarParams) -> Scalar:
        annual_vol_mean = 0.8
        long_term_log_var_mean = 2 * jnp.log(jnp.array(annual_vol_mean))

        stationary_scale = jax.nn.softplus(inference_parameters.sft_inv_stationary_scale)
        half_life_steps = jnp.exp(inference_parameters.log_half_life_steps)
        model_params = self.to_model_parameters(inference_parameters)

        stationary_scale_lp = (
            jstats.norm.logpdf(jnp.log(stationary_scale), loc=jnp.log(0.5), scale=0.35)
            - jnp.log(stationary_scale)
        )

        half_life_steps_lp = (
            jstats.norm.logpdf(jnp.log(half_life_steps), loc=jnp.log(100.0), scale=0.75)
            - jnp.log(half_life_steps)
        )

        long_term_lp = jstats.norm.logpdf(
            model_params.long_term_log_var,
            loc=long_term_log_var_mean,
            scale=0.5,
        )

        return_df_excess = model_params.return_df - 4.0
        return_df_lp = (
            jstats.norm.logpdf(jnp.log(return_df_excess), loc=jnp.log(8.0), scale=0.5)
            - jnp.log(return_df_excess)
        )

        jump_mult_excess = model_params.jump_mult - 2.0
        jump_mult_lp = (
            jstats.norm.logpdf(jnp.log(jump_mult_excess), loc=jnp.log(1.0), scale=0.5)
            - jnp.log(jump_mult_excess)
        )

        lad_stationary_scale = jax.nn.log_sigmoid(inference_parameters.sft_inv_stationary_scale)
        lad_half_life_steps = inference_parameters.log_half_life_steps
        lad_return_df = jax.nn.log_sigmoid(inference_parameters.sft_inv_return_df_m4)
        lad_jump_mult = jax.nn.log_sigmoid(inference_parameters.sft_inv_jump_mult_m2)

        return (
            stationary_scale_lp
            + half_life_steps_lp
            + long_term_lp
            + return_df_lp
            + jump_mult_lp
            + lad_stationary_scale
            + lad_half_life_steps
            + lad_return_df
            + lad_jump_mult
        )

@jax.tree_util.register_dataclass
@dataclass
class JumpStochasticVarBayesian:
    target: typing.ClassVar = JumpStochasticVar()
    parameterization: FullJumpVarParameterization


def jump_var_full(
    hyperparameters: JumpVarHyper = JumpVarHyper(fixed_jump_prob=jnp.array(1e-3, dtype=jnp.float32))
) -> JumpStochasticVarBayesian:
    return JumpStochasticVarBayesian(
        parameterization=FullJumpVarParameterization(hyperparameters=hyperparameters)
    )