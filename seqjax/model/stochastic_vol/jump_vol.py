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

from seqjax.model.typing import Latent, Parameters, NoCondition, NoHyper

from .types import LogReturnObs


prior_order = 2
transition_order = 1
emission_order = 2
observation_dependency = 0

prior_context = partial(LatentContext, length=prior_order)
latent_context = partial(LatentContext, length=transition_order)
observation_context = partial(ObservationContext, length=observation_dependency)
condition_context = partial(ConditionContext, length=0)

latent_context = partial(LatentContext, length=transition_order)
emission_latent_context = partial(LatentContext, length=emission_order)
observation_context = partial(ObservationContext, length=observation_dependency)
condition_context = partial(ConditionContext, length=0)

observation_cls = LogReturnObs
condition_cls = NoCondition


class MicroContamLatent(Latent):
    log_var: Scalar
    micro_noise: Scalar
    _shape_template: typing.ClassVar = OrderedDict(
        log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        micro_noise=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

class MicroContamParams(Parameters):
    std_log_var: Scalar
    ar: Scalar
    long_term_log_var: Scalar

    micro_ar: Scalar
    micro_std: Scalar

    contam_prob: Scalar
    contam_scale: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        micro_ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        micro_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        contam_prob=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        contam_scale=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


def _stationary_log_var_scale(parameters: MicroContamParams) -> Scalar:
    return jnp.sqrt(
        jnp.square(parameters.std_log_var)
        / (1.0 - jnp.square(parameters.ar))
    )


def _stationary_micro_scale(parameters: MicroContamParams) -> Scalar:
    return jnp.sqrt(
        jnp.square(parameters.micro_std)
        / (1.0 - jnp.square(parameters.micro_ar))
    )

def _stationary_log_prob(
    latent: MicroContamLatent,
    parameters: MicroContamParams,
) -> Scalar:
    log_var_sigma = _stationary_log_var_scale(parameters)
    micro_sigma = _stationary_micro_scale(parameters)

    return (
        jstats.norm.logpdf(
            latent.log_var,
            loc=parameters.long_term_log_var,
            scale=log_var_sigma,
        )
        + jstats.norm.logpdf(
            latent.micro_noise,
            loc=0.0,
            scale=micro_sigma,
        )
    )


def _transition_log_prob_single(
    last_latent: MicroContamLatent,
    latent: MicroContamLatent,
    parameters: MicroContamParams,
) -> Scalar:
    log_var_loc = parameters.long_term_log_var + parameters.ar * (
        last_latent.log_var - parameters.long_term_log_var
    )

    micro_loc = parameters.micro_ar * last_latent.micro_noise

    return (
        jstats.norm.logpdf(
            latent.log_var,
            loc=log_var_loc,
            scale=parameters.std_log_var,
        )
        + jstats.norm.logpdf(
            latent.micro_noise,
            loc=micro_loc,
            scale=parameters.micro_std,
        )
    )

def _transition_sample_single(
    key: PRNGKeyArray,
    last_latent: MicroContamLatent,
    parameters: MicroContamParams,
) -> MicroContamLatent:
    log_var_key, micro_key = jrandom.split(key, 2)

    log_var_loc = parameters.long_term_log_var + parameters.ar * (
        last_latent.log_var - parameters.long_term_log_var
    )

    micro_loc = parameters.micro_ar * last_latent.micro_noise

    return MicroContamLatent(
        log_var=log_var_loc + parameters.std_log_var * jrandom.normal(log_var_key),
        micro_noise=micro_loc + parameters.micro_std * jrandom.normal(micro_key),
    )

def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[NoCondition],
    parameters: MicroContamParams,
) -> LatentContext[MicroContamLatent]:
    _ = conditions
    start_key, next_key = jrandom.split(key, 2)

    log_var_key, micro_key = jrandom.split(start_key, 2)

    start_latent = MicroContamLatent(
        log_var=(
            parameters.long_term_log_var
            + _stationary_log_var_scale(parameters) * jrandom.normal(log_var_key)
        ),
        micro_noise=(
            _stationary_micro_scale(parameters) * jrandom.normal(micro_key)
        ),
    )

    next_latent = _transition_sample_single(
        next_key,
        start_latent,
        parameters,
    )

    return prior_context((start_latent, next_latent))

def prior_log_prob(
    latent: LatentContext[MicroContamLatent],
    conditions: ConditionContext[NoCondition],
    parameters: MicroContamParams,
) -> Scalar:
    _ = conditions

    current_latent = latent[0]
    previous_latent = latent[-1]

    return (
        _stationary_log_prob(previous_latent, parameters)
        + _transition_log_prob_single(
            previous_latent,
            current_latent,
            parameters,
        )
    )

def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[MicroContamLatent],
    condition: NoCondition,
    parameters: MicroContamParams,
) -> MicroContamLatent:
    _ = condition
    log_var_key, micro_key = jrandom.split(key, 2)

    last_latent = latent_history[-1]

    log_var_loc = parameters.long_term_log_var + parameters.ar * (
        last_latent.log_var - parameters.long_term_log_var
    )

    micro_loc = parameters.micro_ar * last_latent.micro_noise

    return MicroContamLatent(
        log_var=log_var_loc + parameters.std_log_var * jrandom.normal(log_var_key),
        micro_noise=micro_loc + parameters.micro_std * jrandom.normal(micro_key),
    )


def transition_log_prob(
    latent_history: LatentContext[MicroContamLatent],
    latent: MicroContamLatent,
    condition: NoCondition,
    parameters: MicroContamParams,
) -> Scalar:
    _ = condition

    last_latent = latent_history[-1]

    log_var_loc = parameters.long_term_log_var + parameters.ar * (
        last_latent.log_var - parameters.long_term_log_var
    )

    micro_loc = parameters.micro_ar * last_latent.micro_noise

    return (
        jstats.norm.logpdf(
            latent.log_var,
            loc=log_var_loc,
            scale=parameters.std_log_var,
        )
        + jstats.norm.logpdf(
            latent.micro_noise,
            loc=micro_loc,
            scale=parameters.micro_std,
        )
    )


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[MicroContamLatent],
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: MicroContamParams,
) -> LogReturnObs:
    _ = observation_history
    _ = condition

    return_key, contam_key = jrandom.split(key, 2)

    current_latent = latent_history[-1]
    previous_latent = latent_history[-2]

    return_scale = jnp.exp(0.5 * current_latent.log_var)
    micro_shift = current_latent.micro_noise - previous_latent.micro_noise

    is_contam = jrandom.bernoulli(contam_key, p=parameters.contam_prob)
    return_scale = jnp.where(
        is_contam,
        parameters.contam_scale * return_scale,
        return_scale,
    )

    return LogReturnObs(
        log_return=micro_shift + jrandom.normal(return_key) * return_scale,
    )


def emission_log_prob(
    latent_history: LatentContext[MicroContamLatent],
    observation: LogReturnObs,
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: MicroContamParams,
) -> Scalar:
    _ = observation_history
    _ = condition

    current_latent = latent_history[-1]
    previous_latent = latent_history[-2]

    return_scale = jnp.exp(0.5 * current_latent.log_var)
    micro_shift = current_latent.micro_noise - previous_latent.micro_noise

    base_log_prob = jstats.norm.logpdf(
        observation.log_return,
        loc=micro_shift,
        scale=return_scale,
    )

    contam_log_prob = jstats.norm.logpdf(
        observation.log_return,
        loc=micro_shift,
        scale=parameters.contam_scale * return_scale,
    )

    return jax.scipy.special.logsumexp(
        jnp.stack(
            [
                jnp.log1p(-parameters.contam_prob) + base_log_prob,
                jnp.log(parameters.contam_prob) + contam_log_prob,
            ]
        ),
        axis=0,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MicroContamStochasticVar(
    SequentialModelProtocol[
        MicroContamLatent,
        LogReturnObs,
        NoCondition,
        MicroContamParams,
    ]
):
    prior_order: int = prior_order
    transition_order: int = transition_order
    emission_order: int = emission_order
    observation_dependency: int = observation_dependency

    latent_cls: type[MicroContamLatent] = MicroContamLatent
    observation_cls: type[LogReturnObs] = observation_cls
    parameter_cls: type[MicroContamParams] = MicroContamParams
    condition_cls: type[NoCondition] = condition_cls

    latent_context: typing.Callable[..., LatentContext[MicroContamLatent]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[LogReturnObs]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[NoCondition]] = condition_context

    prior_sample = staticmethod(prior_sample)
    prior_log_prob = staticmethod(prior_log_prob)
    transition_sample = staticmethod(transition_sample)
    transition_log_prob = staticmethod(transition_log_prob)
    emission_sample = staticmethod(emission_sample)
    emission_log_prob = staticmethod(emission_log_prob)


micro_contam_stochastic_var_model = validate_sequential_model(
    MicroContamStochasticVar()
)


class UncMicroContamParams(Parameters):
    sft_inv_std_log_var: Scalar
    logit_ar: Scalar
    long_term_log_var: Scalar

    logit_micro_ar: Scalar
    sft_inv_micro_std: Scalar

    logit_contam_prob: Scalar
    raw_contam_scale: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        logit_ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        sft_inv_std_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        logit_micro_ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        sft_inv_micro_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        logit_contam_prob=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        raw_contam_scale=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MicroContamPriorHyper:
    # Existing SV parameters.
    ar_mean: Scalar = field(default_factory=lambda: jnp.array(0.0))
    ar_std: Scalar = field(default_factory=lambda: jnp.sqrt(jnp.array(1 / 3)))

    long_term_vol_mean: Scalar = field(default_factory=lambda: jnp.array(0.16))
    long_term_vol_std: Scalar = field(default_factory=lambda: jnp.array(0.3))

    std_log_var_mean: Scalar = field(default_factory=lambda: jnp.array(0.2))
    std_log_var_std: Scalar = field(default_factory=lambda: jnp.array(0.1))

    # Microstructure parameters.
    # micro_ar is supported on [0, 1], with mass near small positive values.
    micro_ar_mean: Scalar = field(default_factory=lambda: jnp.array(0.2))
    micro_ar_std: Scalar = field(default_factory=lambda: jnp.array(0.2))

    # This is in the same units as observation.log_return.
    # Set these from spread/tick scale if possible.
    micro_std_mean: Scalar = field(default_factory=lambda: jnp.array(0.02))
    micro_std_std: Scalar = field(default_factory=lambda: jnp.array(0.02))

    # Contamination probability: default mean 0.5 percent per observation.
    contam_prob_mean: Scalar = field(default_factory=lambda: jnp.array(0.005))
    contam_prob_concentration: Scalar = field(default_factory=lambda: jnp.array(200.0))

    # Fixed support for contamination scale.
    contam_scale_min: Scalar = field(default_factory=lambda: jnp.array(3.0))
    contam_scale_max: Scalar = field(default_factory=lambda: jnp.array(8.0))

    @property
    def ar_beta_ab(self):
        m = 0.5 * (self.ar_mean + 1.0)
        var_z = (self.ar_std / 2.0) ** 2
        concentration = m * (1.0 - m) / var_z - 1.0
        return m * concentration, (1.0 - m) * concentration

    @property
    def micro_ar_beta_ab(self):
        m = self.micro_ar_mean
        var_z = self.micro_ar_std**2
        concentration = m * (1.0 - m) / var_z - 1.0
        return m * concentration, (1.0 - m) * concentration

    @property
    def contam_prob_beta_ab(self):
        a = self.contam_prob_mean * self.contam_prob_concentration
        b = (1.0 - self.contam_prob_mean) * self.contam_prob_concentration
        return a, b

    @property
    def long_term_log_var_mean_std(self):
        cv2 = jnp.square(self.long_term_vol_std / self.long_term_vol_mean)

        log_vol_sd = jnp.sqrt(jnp.log1p(cv2))
        log_vol_loc = (
            jnp.log(self.long_term_vol_mean)
            - 0.5 * jnp.square(log_vol_sd)
        )

        long_term_log_var_mean = 2.0 * log_vol_loc
        long_term_log_var_sd = 2.0 * log_vol_sd

        return long_term_log_var_mean, long_term_log_var_sd

    @property
    def std_log_var_log_mean_std(self):
        cv2 = jnp.square(self.std_log_var_std / self.std_log_var_mean)

        log_sd = jnp.sqrt(jnp.log1p(cv2))
        log_loc = (
            jnp.log(self.std_log_var_mean)
            - 0.5 * jnp.square(log_sd)
        )

        return log_loc, log_sd

    @property
    def micro_std_log_mean_std(self):
        cv2 = jnp.square(self.micro_std_std / self.micro_std_mean)

        log_sd = jnp.sqrt(jnp.log1p(cv2))
        log_loc = (
            jnp.log(self.micro_std_mean)
            - 0.5 * jnp.square(log_sd)
        )

        return log_loc, log_sd


def _logit(x):
    return jnp.log(x) - jnp.log1p(-x)


def _bounded_log_contam_scale(raw_contam_scale, lo, hi):
    z = jax.nn.sigmoid(raw_contam_scale)
    return jnp.log(lo) + (jnp.log(hi) - jnp.log(lo)) * z


@jax.tree_util.register_dataclass
@dataclass
class MicroContamParameterization(
    ParameterizationProtocol[
        MicroContamParams,
        UncMicroContamParams,
        MicroContamPriorHyper,
    ]
):
    _hyperparameters: MicroContamPriorHyper = field(
        default_factory=MicroContamPriorHyper
    )
    inference_parameter_cls: typing.ClassVar[type[UncMicroContamParams]] = (
        UncMicroContamParams
    )

    @property
    def hyperparameters(self):
        return jax.lax.stop_gradient(self._hyperparameters)

    def to_model_parameters(
        self,
        inference_parameters: UncMicroContamParams,
    ) -> MicroContamParams:
        log_contam_scale = _bounded_log_contam_scale(
            inference_parameters.raw_contam_scale,
            self.hyperparameters.contam_scale_min,
            self.hyperparameters.contam_scale_max,
        )

        return MicroContamParams(
            std_log_var=jax.nn.softplus(
                inference_parameters.sft_inv_std_log_var
            ),
            ar=jnp.tanh(inference_parameters.logit_ar),
            long_term_log_var=inference_parameters.long_term_log_var,

            micro_ar=jax.nn.sigmoid(inference_parameters.logit_micro_ar),
            micro_std=jax.nn.softplus(
                inference_parameters.sft_inv_micro_std
            ),

            contam_prob=jax.nn.sigmoid(
                inference_parameters.logit_contam_prob
            ),
            contam_scale=jnp.exp(log_contam_scale),
        )

    def from_model_parameters(
        self,
        model_parameters: MicroContamParams,
    ) -> UncMicroContamParams:
        lo = self.hyperparameters.contam_scale_min
        hi = self.hyperparameters.contam_scale_max

        log_scale01 = (
            (jnp.log(model_parameters.contam_scale) - jnp.log(lo))
            / (jnp.log(hi) - jnp.log(lo))
        )

        return UncMicroContamParams(
            sft_inv_std_log_var=jnp.log(
                jnp.expm1(model_parameters.std_log_var)
            ),
            logit_ar=jnp.arctanh(model_parameters.ar),
            long_term_log_var=model_parameters.long_term_log_var,

            logit_micro_ar=_logit(model_parameters.micro_ar),
            sft_inv_micro_std=jnp.log(
                jnp.expm1(model_parameters.micro_std)
            ),

            logit_contam_prob=_logit(model_parameters.contam_prob),
            raw_contam_scale=_logit(log_scale01),
        )

    def sample(self, key: PRNGKeyArray) -> UncMicroContamParams:
        k1, k2, k3, k4, k5, k6, k7 = jrandom.split(key, 7)

        ar_beta_a, ar_beta_b = self.hyperparameters.ar_beta_ab
        micro_ar_beta_a, micro_ar_beta_b = self.hyperparameters.micro_ar_beta_ab
        contam_prob_beta_a, contam_prob_beta_b = (
            self.hyperparameters.contam_prob_beta_ab
        )

        long_term_log_var_mean, long_term_log_var_std = (
            self.hyperparameters.long_term_log_var_mean_std
        )
        std_log_var_mean, std_log_var_std = (
            self.hyperparameters.std_log_var_log_mean_std
        )
        micro_std_mean, micro_std_std = (
            self.hyperparameters.micro_std_log_mean_std
        )

        lo = self.hyperparameters.contam_scale_min
        hi = self.hyperparameters.contam_scale_max

        # Log-uniform prior on contam_scale in [lo, hi].
        contam_scale01 = jrandom.uniform(k7)
        log_contam_scale = (
            jnp.log(lo)
            + (jnp.log(hi) - jnp.log(lo)) * contam_scale01
        )

        return self.from_model_parameters(
            MicroContamParams(
                std_log_var=jnp.exp(
                    std_log_var_mean
                    + std_log_var_std * jrandom.normal(k1)
                ),
                ar=2.0 * jrandom.beta(k2, ar_beta_a, ar_beta_b) - 1.0,
                long_term_log_var=(
                    long_term_log_var_mean
                    + long_term_log_var_std * jrandom.normal(k3)
                ),

                micro_ar=jrandom.beta(
                    k4,
                    micro_ar_beta_a,
                    micro_ar_beta_b,
                ),
                micro_std=jnp.exp(
                    micro_std_mean
                    + micro_std_std * jrandom.normal(k5)
                ),

                contam_prob=jrandom.beta(
                    k6,
                    contam_prob_beta_a,
                    contam_prob_beta_b,
                ),
                contam_scale=jnp.exp(log_contam_scale),
            )
        )

    def log_prob(self, inference_parameters: UncMicroContamParams) -> Scalar:
        model_params = self.to_model_parameters(inference_parameters)

        # std_log_var prior and Jacobian.
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

        # long_term_log_var prior.
        long_term_log_var_mean, long_term_log_var_std = (
            self.hyperparameters.long_term_log_var_mean_std
        )
        long_term_lp = jstats.norm.logpdf(
            model_params.long_term_log_var,
            loc=long_term_log_var_mean,
            scale=long_term_log_var_std,
        )

        # log_var AR prior and Jacobian.
        ar01 = 0.5 * (model_params.ar + 1.0)
        ar_beta_a, ar_beta_b = self.hyperparameters.ar_beta_ab
        ar_lp = (
            jstats.beta.logpdf(ar01, a=ar_beta_a, b=ar_beta_b)
            - jnp.log(2.0)
        )
        lad_ar = jnp.log1p(-jnp.square(model_params.ar))

        # micro_ar prior and Jacobian.
        micro_ar_beta_a, micro_ar_beta_b = self.hyperparameters.micro_ar_beta_ab
        micro_ar_lp = jstats.beta.logpdf(
            model_params.micro_ar,
            a=micro_ar_beta_a,
            b=micro_ar_beta_b,
        )
        lad_micro_ar = (
            jax.nn.log_sigmoid(inference_parameters.logit_micro_ar)
            + jax.nn.log_sigmoid(-inference_parameters.logit_micro_ar)
        )

        # micro_std prior and Jacobian.
        micro_std_mean, micro_std_std = self.hyperparameters.micro_std_log_mean_std
        micro_std_lp = (
            jstats.norm.logpdf(
                jnp.log(model_params.micro_std),
                loc=micro_std_mean,
                scale=micro_std_std,
            )
            - jnp.log(model_params.micro_std)
        )
        lad_micro_std = jax.nn.log_sigmoid(
            inference_parameters.sft_inv_micro_std
        )

        # contam_prob prior and Jacobian.
        contam_prob_beta_a, contam_prob_beta_b = (
            self.hyperparameters.contam_prob_beta_ab
        )
        contam_prob_lp = jstats.beta.logpdf(
            model_params.contam_prob,
            a=contam_prob_beta_a,
            b=contam_prob_beta_b,
        )
        lad_contam_prob = (
            jax.nn.log_sigmoid(inference_parameters.logit_contam_prob)
            + jax.nn.log_sigmoid(-inference_parameters.logit_contam_prob)
        )

        # contam_scale prior:
        # log-uniform over [3, 8] by default.
        # Since contam_scale = exp(log_lo + width * sigmoid(raw)),
        # the density in raw space reduces to the logistic Jacobian.
        lad_contam_scale = (
            jax.nn.log_sigmoid(inference_parameters.raw_contam_scale)
            + jax.nn.log_sigmoid(-inference_parameters.raw_contam_scale)
        )

        return (
            std_lp
            + ar_lp
            + long_term_lp
            + lad_ar
            + lad_std_log_var
            + micro_ar_lp
            + lad_micro_ar
            + micro_std_lp
            + lad_micro_std
            + contam_prob_lp
            + lad_contam_prob
            + lad_contam_scale
        )


@jax.tree_util.register_dataclass
@dataclass
class MicroContamStochasticVarBayesian:
    target: typing.ClassVar = micro_contam_stochastic_var_model
    parameterization: MicroContamParameterization


def svar_micro_contam(
    hyperparameters: MicroContamPriorHyper = MicroContamPriorHyper(),
) -> MicroContamStochasticVarBayesian:
    return MicroContamStochasticVarBayesian(
        parameterization=MicroContamParameterization(hyperparameters)
    )