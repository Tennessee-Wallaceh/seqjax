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
#     dh_t = mean_reversion_rate * (long_term_log_var - h_t) dt
#            + std_log_var dW_t.
#
# Parameter units:
#   mean_reversion_rate: inverse trading years.
#   std_log_var:         log-variance units per square-root trading year.
#   long_term_log_var:   log annualised variance.


class TimeStepCondition(Condition):
    timestep: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        timestep=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class LogVarParams(Parameters):
    std_log_var: Scalar
    mean_reversion_rate: Scalar
    long_term_log_var: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        mean_reversion_rate=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
        long_term_log_var=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
    )


prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0

latent_cls = LatentVar
observation_cls = LogReturnObs
parameter_cls = LogVarParams
condition_cls = TimeStepCondition

latent_context = partial(LatentContext, length=transition_order)
observation_context = partial(
    ObservationContext,
    length=observation_dependency,
)
condition_context = partial(ConditionContext, length=0)


# This gives annualised volatility exp(h_0 / 2) a central 95% interval of
# exactly 5% to 50%, while retaining full support for h_0.
initial_vol_lower = 0.05
initial_vol_upper = 0.50
standard_normal_975 = 1.959963984540054

initial_log_var_mean = jnp.log(
    initial_vol_lower * initial_vol_upper
)
initial_log_var_std = (
    jnp.log(initial_vol_upper / initial_vol_lower)
    / standard_normal_975
)


def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[TimeStepCondition],
    parameters: LogVarParams,
) -> LatentContext[LatentVar]:
    _ = conditions
    _ = parameters

    start_lv = LatentVar(
        log_var=(
            initial_log_var_mean
            + initial_log_var_std * jrandom.normal(key)
        )
    )
    return latent_context((start_lv,))


def prior_log_prob(
    latent: LatentContext[LatentVar],
    conditions: ConditionContext[TimeStepCondition],
    parameters: LogVarParams,
) -> Scalar:
    _ = conditions
    _ = parameters

    return jstats.norm.logpdf(
        latent[0].log_var,
        loc=initial_log_var_mean,
        scale=initial_log_var_std,
    )


def _stationary_scale(parameters: LogVarParams) -> Scalar:
    return parameters.std_log_var / jnp.sqrt(
        2.0 * parameters.mean_reversion_rate
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
    condition: TimeStepCondition,
    parameters: LogVarParams,
) -> LatentVar:
    last_log_var = latent_history[0]
    decay = _transition_decay(condition.timestep, parameters)
    loc = parameters.long_term_log_var + decay * (
        last_log_var.log_var - parameters.long_term_log_var
    )
    scale = _transition_scale(condition.timestep, parameters)
    return LatentVar(
        log_var=loc + scale * jrandom.normal(key)
    )


def transition_log_prob(
    latent_history: LatentContext[LatentVar],
    latent: LatentVar,
    condition: TimeStepCondition,
    parameters: LogVarParams,
) -> Scalar:
    last_log_var = latent_history[0]
    decay = _transition_decay(condition.timestep, parameters)
    loc = parameters.long_term_log_var + decay * (
        last_log_var.log_var - parameters.long_term_log_var
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
    observation_history: ObservationContext[LogReturnObs],
    condition: TimeStepCondition,
    parameters: LogVarParams,
) -> LogReturnObs:
    _ = observation_history
    _ = parameters
    current_latent = latent_history[0]

    # The observation is the raw log return over condition.timestep; the
    # latent variance is annualised.
    return_scale = jnp.sqrt(condition.timestep) * jnp.exp(
        0.5 * current_latent.log_var
    )
    return LogReturnObs(
        log_return=jrandom.normal(key) * return_scale
    )


def emission_log_prob(
    latent_history: LatentContext[LatentVar],
    observation: LogReturnObs,
    observation_history: ObservationContext[LogReturnObs],
    condition: TimeStepCondition,
    parameters: LogVarParams,
) -> Scalar:
    _ = observation_history
    _ = parameters
    current_latent = latent_history[0]
    return_scale = jnp.sqrt(condition.timestep) * jnp.exp(
        0.5 * current_latent.log_var
    )
    return jstats.norm.logpdf(
        observation.log_return,
        loc=0.0,
        scale=return_scale,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SimpleStochasticVar(
    SequentialModelProtocol[
        LatentVar,
        LogReturnObs,
        TimeStepCondition,
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
    condition_cls: type[TimeStepCondition] = condition_cls

    latent_context: typing.Callable[..., LatentContext[LatentVar]] = (
        latent_context
    )
    observation_context: typing.Callable[
        ..., ObservationContext[LogReturnObs]
    ] = observation_context
    condition_context: typing.Callable[
        ..., ConditionContext[TimeStepCondition]
    ] = condition_context

    prior_sample = staticmethod(prior_sample)
    prior_log_prob = staticmethod(prior_log_prob)
    transition_sample = staticmethod(transition_sample)
    transition_log_prob = staticmethod(transition_log_prob)
    emission_sample = staticmethod(emission_sample)
    emission_log_prob = staticmethod(emission_log_prob)


simple_stochastic_var_model = validate_sequential_model(
    SimpleStochasticVar()
)


class UncLogVarParams(Parameters):
    sft_inv_std_log_var: Scalar
    sft_inv_mean_reversion_rate: Scalar
    long_term_log_var: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        sft_inv_std_log_var=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
        sft_inv_mean_reversion_rate=jax.ShapeDtypeStruct(
            shape=(),
            dtype=jnp.float32,
        ),
        long_term_log_var=jax.ShapeDtypeStruct(
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
        default_factory=lambda: jnp.array(10.0)
    )
    mean_reversion_rate_std: Scalar = field(
        default_factory=lambda: jnp.array(5.0)
    )

    # Log-normal prior on exp(long_term_log_var / 2).
    long_term_vol_mean: Scalar = field(
        default_factory=lambda: jnp.array(0.16)
    )
    long_term_vol_std: Scalar = field(
        default_factory=lambda: jnp.array(0.10)
    )

    # Log-normal prior on annualised log-variance diffusion.
    std_log_var_mean: Scalar = field(
        default_factory=lambda: jnp.array(4.0)
    )
    std_log_var_std: Scalar = field(
        default_factory=lambda: jnp.array(2.0)
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
    def long_term_log_var_mean_std(
        self,
    ) -> tuple[Scalar, Scalar]:
        log_vol_mean, log_vol_std = self._lognormal_log_mean_std(
            self.long_term_vol_mean,
            self.long_term_vol_std,
        )
        return 2.0 * log_vol_mean, 2.0 * log_vol_std

    @property
    def std_log_var_log_mean_std(
        self,
    ) -> tuple[Scalar, Scalar]:
        return self._lognormal_log_mean_std(
            self.std_log_var_mean,
            self.std_log_var_std,
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
            long_term_log_var=inference_parameters.long_term_log_var,
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
            long_term_log_var=model_parameters.long_term_log_var,
        )

    def sample(self, key: PRNGKeyArray) -> UncLogVarParams:
        k1, k2, k3 = jrandom.split(key, 3)

        std_log_var_mean, std_log_var_std = (
            self.hyperparameters.std_log_var_log_mean_std
        )
        mean_reversion_rate_mean, mean_reversion_rate_std = (
            self.hyperparameters.mean_reversion_rate_log_mean_std
        )
        long_term_log_var_mean, long_term_log_var_std = (
            self.hyperparameters.long_term_log_var_mean_std
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
                long_term_log_var=(
                    long_term_log_var_mean
                    + long_term_log_var_std * jrandom.normal(k3)
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

        long_term_log_var_mean, long_term_log_var_std = (
            self.hyperparameters.long_term_log_var_mean_std
        )
        long_term_lp = jstats.norm.logpdf(
            model_params.long_term_log_var,
            loc=long_term_log_var_mean,
            scale=long_term_log_var_std,
        )

        return (
            std_lp
            + mean_reversion_rate_lp
            + long_term_lp
            + lad_std_log_var
            + lad_mean_reversion_rate
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