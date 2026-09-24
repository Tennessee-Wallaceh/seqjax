from collections import OrderedDict
from dataclasses import dataclass, field
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
    ParameterizationProtocol,
)
from seqjax.model.typing import Condition, Parameters

from .types import LatentVar, LogReturnObs


# Time is measured in trading years. For 256 trading days, each containing
# 8 * 60 minutes, a one-minute timestep is 1 / (256 * 8 * 60).
#
# The latent state is log annualised variance:
#
#     h_t = log(sigma_t ** 2).
#
# Its continuous-time dynamics are
#
#     dh_t = mean_reversion_rate * (stationary_log_var_mean - h_t) dt
#            + std_log_var dW_t.
#
# ``long_term_log_vol`` is parameterised as
#
#     log(E[exp(h_infinity / 2)]).
#
# Consequently, exp(long_term_log_vol) is the long-run mean annualised
# conditional volatility, E[sqrt(V_infinity)], rather than its median.
# The OU reversion centre is adjusted below for the stationary
# log-variance variance.
#
# Parameter units:
#   mean_reversion_rate: inverse trading years.
#   std_log_var:         log-variance units per square-root trading year.
#   long_term_log_vol:   log stationary mean annualised volatility.
#   df:                  Student-t degrees of freedom.


minimum_df = 2.5


class TimeStepCondition(Condition):
    """Elapsed trading time associated with one model step."""

    timestep: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        timestep=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarParams(Parameters):
    std_log_var: Scalar
    mean_reversion_rate: Scalar
    long_term_log_vol: Scalar
    df: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        mean_reversion_rate=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
        long_term_log_vol=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
        df=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
    )


transition_latent_order = 1
emission_latent_order = 1
transition_observation_order = 0
emission_observation_order = 0

latent_cls = LatentVar
observation_cls = LogReturnObs
parameter_cls = LogVarParams
condition_cls = TimeStepCondition

def _stationary_scale(parameters: LogVarParams) -> Scalar:
    return parameters.std_log_var / jnp.sqrt(
        2.0 * parameters.mean_reversion_rate
    )


def _stationary_log_var_mean(
    parameters: LogVarParams,
) -> Scalar:
    # If h_infinity ~ Normal(mu_h, stationary_scale**2), then
    #
    # E[exp(h_infinity / 2)]
    #     = exp(mu_h / 2 + stationary_scale**2 / 8).
    #
    # This correction therefore makes
    # exp(long_term_log_vol) = E[exp(h_infinity / 2)].
    stationary_variance = jnp.square(
        _stationary_scale(parameters)
    )
    return (
        2.0 * parameters.long_term_log_vol
        - 0.25 * stationary_variance
    )


def prior_sample(
    key: PRNGKeyArray,
    parameters: LogVarParams,
) -> LatentContext[LatentVar]:

    start_lv = LatentVar(
        log_var=(
            _stationary_log_var_mean(parameters)
            + _stationary_scale(parameters) * jrandom.normal(key)
        )
    )
    return LatentContext.from_values(start_lv, length=max(transition_latent_order, emission_latent_order))


def prior_log_prob(
    latent: LatentContext[LatentVar],
    parameters: LogVarParams,
) -> Scalar:

    return jstats.norm.logpdf(
        latent[-1].log_var,
        loc=_stationary_log_var_mean(parameters),
        scale=_stationary_scale(parameters),
    )


def _transition_decay(
    timestep: Scalar,
    parameters: LogVarParams,
) -> Scalar:
    return jnp.exp(
        -parameters.mean_reversion_rate * timestep
    )


def _transition_scale(
    timestep: Scalar,
    parameters: LogVarParams,
) -> Scalar:
    mean_reversion_rate = parameters.mean_reversion_rate

    # Exact OU transition variance. The expm1 form is accurate when the
    # timestep is small. As mean_reversion_rate approaches zero, this tends to
    # std_log_var * sqrt(timestep).
    transition_variance = (
        -jnp.expm1(-2.0 * mean_reversion_rate * timestep)
        / (2.0 * mean_reversion_rate)
    )
    return parameters.std_log_var * jnp.sqrt(transition_variance)


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVar],
    parameters: LogVarParams,
    condition: TimeStepCondition,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeStepCondition],
) -> LatentVar:
    _ = observation_history
    last_log_var = latent_history[-1]
    decay = _transition_decay(condition.timestep, parameters)
    stationary_log_var_mean = _stationary_log_var_mean(parameters)
    loc = stationary_log_var_mean + decay * (
        last_log_var.log_var - stationary_log_var_mean
    )
    scale = _transition_scale(condition.timestep, parameters)
    return LatentVar(
        log_var=loc + scale * jrandom.normal(key)
    )


def transition_log_prob(
    latent_history: LatentContext[LatentVar],
    latent: LatentVar,
    parameters: LogVarParams,
    condition: TimeStepCondition,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeStepCondition],
) -> Scalar:
    _ = observation_history
    last_log_var = latent_history[-1]
    decay = _transition_decay(condition.timestep, parameters)
    stationary_log_var_mean = _stationary_log_var_mean(parameters)
    loc = stationary_log_var_mean + decay * (
        last_log_var.log_var - stationary_log_var_mean
    )
    scale = _transition_scale(condition.timestep, parameters)
    return jstats.norm.logpdf(
        latent.log_var,
        loc=loc,
        scale=scale,
    )


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentVar],
    parameters: LogVarParams,
    condition: TimeStepCondition,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeStepCondition],
) -> LogReturnObs:
    _ = observation_history
    current_latent = latent_history[-1]

    # jrandom.t has variance df / (df - 2). Multiplying its scale by
    # sqrt((df - 2) / df) makes exp(log_var) remain the conditional
    # annualised return variance.
    return_scale = (
        jnp.sqrt(condition.timestep)
        * jnp.exp(0.5 * current_latent.log_var)
        * jnp.sqrt((parameters.df - 2.0) / parameters.df)
    )
    return LogReturnObs(
        log_return=(
            jrandom.t(key, parameters.df) * return_scale
        )
    )


def emission_log_prob(
    latent_history: LatentContext[LatentVar],
    observation: LogReturnObs,
    parameters: LogVarParams,
    condition: TimeStepCondition,
    observation_history: ObservedHistoryContext[LogReturnObs, TimeStepCondition],
) -> Scalar:
    _ = observation_history
    current_latent = latent_history[-1]
    return_scale = (
        jnp.sqrt(condition.timestep)
        * jnp.exp(0.5 * current_latent.log_var)
        * jnp.sqrt((parameters.df - 2.0) / parameters.df)
    )
    return jstats.t.logpdf(
        observation.log_return,
        df=parameters.df,
        loc=0.0,
        scale=return_scale,
    )


simple_stochastic_var_model = SequentialModel(
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


def SimpleStochasticVar() -> SequentialModel:
    """Construct the time-varying stochastic-variance model."""

    return simple_stochastic_var_model


class UncLogVarParams(Parameters):
    sft_inv_std_log_var: Scalar
    sft_inv_mean_reversion_rate: Scalar
    long_term_log_vol: Scalar
    sft_inv_df: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        sft_inv_std_log_var=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
        sft_inv_mean_reversion_rate=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
        long_term_log_vol=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
        sft_inv_df=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LogVarPriorHyper:
    # Log-normal prior on the annual mean-reversion rate. These defaults put
    # most prior mass on half-lives of roughly 8 to 50 trading days.
    mean_reversion_rate_mean: Scalar = field(
        default_factory=lambda: jnp.array(15.0)
    )
    mean_reversion_rate_std: Scalar = field(
        default_factory=lambda: jnp.array(15.0)
    )


    # Log-normal prior on the long-run mean conditional volatility
    # exp(long_term_log_vol) = E[sqrt(V_infinity)].
    long_term_vol_mean: Scalar = field(
        default_factory=lambda: jnp.array(0.16)
    )
    long_term_vol_std: Scalar = field(
        default_factory=lambda: jnp.array(0.10)
    )

    # Log-normal prior on annualised log-variance diffusion.
    std_log_var_mean: Scalar = field(
        default_factory=lambda: jnp.array(6.0)
    )
    std_log_var_std: Scalar = field(
        default_factory=lambda: jnp.array(5.0)
    )

    # Log-normal prior on df - minimum_df. These defaults imply a prior
    # mean of 10 for df while retaining support arbitrarily close to the
    # lower bound.
    df_excess_mean: Scalar = field(
        default_factory=lambda: jnp.array(7.5)
    )
    df_excess_std: Scalar = field(
        default_factory=lambda: jnp.array(5.0)
    )

    @staticmethod
    def _lognormal_log_mean_std(
        mean: Scalar,
        std: Scalar,
    ) -> tuple[Scalar, Scalar]:
        cv2 = jnp.square(std / mean)
        log_std = jnp.sqrt(jnp.log1p(cv2))
        log_mean = jnp.log(mean) - 0.5 * jnp.square(log_std)
        return log_mean, log_std

    @property
    def mean_reversion_rate_log_mean_std(
        self,
    ) -> tuple[Scalar, Scalar]:
        return self._lognormal_log_mean_std(
            self.mean_reversion_rate_mean,
            self.mean_reversion_rate_std,
        )

    @property
    def long_term_log_vol_mean_std(
        self,
    ) -> tuple[Scalar, Scalar]:
        return self._lognormal_log_mean_std(
            self.long_term_vol_mean,
            self.long_term_vol_std,
        )

    @property
    def std_log_var_log_mean_std(
        self,
    ) -> tuple[Scalar, Scalar]:
        return self._lognormal_log_mean_std(
            self.std_log_var_mean,
            self.std_log_var_std,
        )

    @property
    def df_excess_log_mean_std(
        self,
    ) -> tuple[Scalar, Scalar]:
        return self._lognormal_log_mean_std(
            self.df_excess_mean,
            self.df_excess_std,
        )


def _softplus_inverse(value: Scalar) -> Scalar:
    # Equivalent to log(expm1(value)), but stable for large value.
    return value + jnp.log(-jnp.expm1(-value))


@jax.tree_util.register_dataclass
@dataclass
class FullVarParameterization(
    ParameterizationProtocol[
        LogVarParams,
        UncLogVarParams,
        LogVarPriorHyper,
    ]
):
    _hyperparameters: LogVarPriorHyper = field(
        default_factory=LogVarPriorHyper
    )
    inference_parameter_cls: typing.ClassVar[type[UncLogVarParams]] = (
        UncLogVarParams
    )

    @property
    def hyperparameters(self) -> LogVarPriorHyper:
        return jax.lax.stop_gradient(self._hyperparameters)

    def to_model_parameters(
        self,
        inference_parameters: UncLogVarParams,
    ) -> LogVarParams:
        return LogVarParams(
            std_log_var=jax.nn.softplus(
                inference_parameters.sft_inv_std_log_var
            ),
            mean_reversion_rate=jax.nn.softplus(
                inference_parameters.sft_inv_mean_reversion_rate
            ),
            long_term_log_vol=inference_parameters.long_term_log_vol,
            df=(
                minimum_df
                + jax.nn.softplus(
                    inference_parameters.sft_inv_df
                )
            ),
        )

    def from_model_parameters(
        self,
        model_parameters: LogVarParams,
    ) -> UncLogVarParams:
        return UncLogVarParams(
            sft_inv_std_log_var=_softplus_inverse(
                model_parameters.std_log_var
            ),
            sft_inv_mean_reversion_rate=_softplus_inverse(
                model_parameters.mean_reversion_rate
            ),
            long_term_log_vol=model_parameters.long_term_log_vol,
            sft_inv_df=_softplus_inverse(
                model_parameters.df - minimum_df
            ),
        )

    def sample(self, key: PRNGKeyArray) -> UncLogVarParams:
        k1, k2, k3, k4 = jrandom.split(key, 4)

        std_log_var_mean, std_log_var_std = (
            self.hyperparameters.std_log_var_log_mean_std
        )
        mean_reversion_rate_mean, mean_reversion_rate_std = (
            self.hyperparameters.mean_reversion_rate_log_mean_std
        )
        long_term_log_vol_mean, long_term_log_vol_std = (
            self.hyperparameters.long_term_log_vol_mean_std
        )
        df_excess_mean, df_excess_std = (
            self.hyperparameters.df_excess_log_mean_std
        )

        return self.from_model_parameters(
            LogVarParams(
                std_log_var=jnp.exp(
                    std_log_var_mean
                    + std_log_var_std * jrandom.normal(k1)
                ),
                mean_reversion_rate=jnp.exp(
                    mean_reversion_rate_mean
                    + mean_reversion_rate_std * jrandom.normal(k2)
                ),
                long_term_log_vol=(
                    long_term_log_vol_mean
                    + long_term_log_vol_std * jrandom.normal(k3)
                ),
                df=(
                    minimum_df
                    + jnp.exp(
                        df_excess_mean
                        + df_excess_std * jrandom.normal(k4)
                    )
                ),
            )
        )

    def log_prob(
        self,
        inference_parameters: UncLogVarParams,
    ) -> Scalar:
        model_params = self.to_model_parameters(inference_parameters)

        std_log_var_mean, std_log_var_std = (
            self.hyperparameters.std_log_var_log_mean_std
        )
        std_lp = (
            jstats.norm.logpdf(
                jnp.log(model_params.std_log_var),
                loc=std_log_var_mean,
                scale=std_log_var_std,
            )
            - jnp.log(model_params.std_log_var)
        )
        lad_std_log_var = jax.nn.log_sigmoid(
            inference_parameters.sft_inv_std_log_var
        )

        mean_reversion_rate_mean, mean_reversion_rate_std = (
            self.hyperparameters.mean_reversion_rate_log_mean_std
        )
        mean_reversion_rate_lp = (
            jstats.norm.logpdf(
                jnp.log(model_params.mean_reversion_rate),
                loc=mean_reversion_rate_mean,
                scale=mean_reversion_rate_std,
            )
            - jnp.log(model_params.mean_reversion_rate)
        )
        lad_mean_reversion_rate = jax.nn.log_sigmoid(
            inference_parameters.sft_inv_mean_reversion_rate
        )

        long_term_log_vol_mean, long_term_log_vol_std = (
            self.hyperparameters.long_term_log_vol_mean_std
        )
        long_term_lp = jstats.norm.logpdf(
            model_params.long_term_log_vol,
            loc=long_term_log_vol_mean,
            scale=long_term_log_vol_std,
        )

        df_excess_mean, df_excess_std = (
            self.hyperparameters.df_excess_log_mean_std
        )
        df_excess = model_params.df - minimum_df
        df_lp = (
            jstats.norm.logpdf(
                jnp.log(df_excess),
                loc=df_excess_mean,
                scale=df_excess_std,
            )
            - jnp.log(df_excess)
        )
        lad_df = jax.nn.log_sigmoid(
            inference_parameters.sft_inv_df
        )

        return (
            std_lp
            + mean_reversion_rate_lp
            + long_term_lp
            + df_lp
            + lad_std_log_var
            + lad_mean_reversion_rate
            + lad_df
        )


@jax.tree_util.register_dataclass
@dataclass
class StochasticVarBayesian:
    target: typing.ClassVar = simple_stochastic_var_model
    parameterization: FullVarParameterization


def svar_full(
    hyperparameters: LogVarPriorHyper = LogVarPriorHyper(),
) -> StochasticVarBayesian:
    return StochasticVarBayesian(
        parameterization=FullVarParameterization(hyperparameters)
    )
