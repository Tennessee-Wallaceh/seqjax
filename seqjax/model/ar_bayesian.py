from dataclasses import field, dataclass
from . import ar_model, ar as ar_module
import jax.numpy as jnp
import jax.scipy.stats as jstats
from seqjax.model.typing import Parameters
from jaxtyping import Scalar, PRNGKeyArray
import jax.random as jrandom
import jax
import typing
from collections import OrderedDict
from .interface import ParameterizationProtocol
import seqjax.model.typing as seqjtyping

class UncARParameters(Parameters):
    logit_ar: Scalar
    sft_inv_obs_std: Scalar
    sft_inv_trns_std: Scalar
    _shape_template: typing.ClassVar = OrderedDict(
        logit_ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        sft_inv_obs_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        sft_inv_trns_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

class UncAROnlyParameters(Parameters):
    logit_ar: Scalar = field(default_factory=lambda: jnp.array(0.0))
    _shape_template: typing.ClassVar = OrderedDict(
        logit_ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


def ar_parameter_sample(key: PRNGKeyArray) -> Scalar:
    return jrandom.uniform(key, minval=-1, maxval=1)

def ar_parameter_log_prob(
    ar_param: Scalar
) -> Scalar:
    log_p_theta = jstats.uniform.logpdf(ar_param, loc=-1.0, scale=2.0)
    return log_p_theta

def obs_std_sample(key: PRNGKeyArray) -> Scalar:
    return jnp.abs(jrandom.cauchy(key))

def obs_std_log_prob(
    observation_std: Scalar
) -> Scalar:
    log_2 = jnp.log(jnp.array(2.0))
    log_p_theta = jstats.cauchy.logpdf(observation_std) + log_2
    return log_p_theta

def trns_parameter_sample(key: PRNGKeyArray) -> Scalar:
    return jnp.abs(jrandom.cauchy(key))

def trns_parameter_log_prob(
    transition_std: Scalar
) -> Scalar:
    log_2 = jnp.log(jnp.array(2.0))
    log_p_theta = jstats.cauchy.logpdf(transition_std) + log_2
    return log_p_theta


@dataclass(frozen=True)
class FixedARParams:
    fixed_observation_std: Scalar
    fixed_transition_std: Scalar

@dataclass
class AROnlyParameterization(
    ParameterizationProtocol[
        ar_module.ARParameters, 
        UncAROnlyParameters,
        FixedARParams,
    ]
):
    inference_parameter_cls = UncAROnlyParameters
    hyperparameters: FixedARParams

    def to_model_parameters(
        self,
        inference_parameters: UncAROnlyParameters,
    ) -> ar_module.ARParameters:
        return ar_module.ARParameters(
            ar=jax.nn.tanh(inference_parameters.logit_ar),
            observation_std=self.hyperparameters.fixed_observation_std,
            transition_std=self.hyperparameters.fixed_transition_std,
        )

    def from_model_parameters(
        self,
        model_parameters: ar_module.ARParameters,
    ) -> UncAROnlyParameters:
        return UncAROnlyParameters(
            logit_ar=jnp.arctanh(model_parameters.ar)
        )
    

    def sample(self, key):
        return UncAROnlyParameters(
            logit_ar=jnp.arctanh(ar_parameter_sample(key))
        )
    
    def log_prob(self, inference_parameters: UncAROnlyParameters) -> Scalar:
        x = inference_parameters.logit_ar
        ar = jnp.tanh(x)
        lad = jnp.log(4.0) + jax.nn.log_sigmoid(2.0 * x) + jax.nn.log_sigmoid(-2.0 * x)
        return ar_parameter_log_prob(ar) + lad

@dataclass
class AROnlyBayesian:
    target: typing.ClassVar = ar_model
    parameterization: AROnlyParameterization

def ar_only(hyperparameters: FixedARParams) -> AROnlyBayesian:
    return AROnlyBayesian(
        parameterization=AROnlyParameterization(hyperparameters)
    )


@dataclass
class FullParameterization(
    ParameterizationProtocol[
        ar_module.ARParameters,
        UncARParameters,
        seqjtyping.NoHyper,
    ]
):
    inference_parameter_cls: typing.ClassVar[type[UncARParameters]] = UncARParameters
    hyperparameters: seqjtyping.NoHyper = field(default_factory=seqjtyping.NoHyper)

    def to_model_parameters(
        self,
        inference_parameters: UncARParameters,
    ) -> ar_module.ARParameters:
        ar_param = jnp.tanh(inference_parameters.logit_ar)
        observation_std = jax.nn.softplus(inference_parameters.sft_inv_obs_std)
        transition_std = jax.nn.softplus(inference_parameters.sft_inv_trns_std)

        return ar_module.ARParameters(
            ar=ar_param,
            observation_std=observation_std,
            transition_std=transition_std,
        )

    def from_model_parameters(
        self,
        model_parameters: ar_module.ARParameters,
    ) -> UncARParameters:
        eps_ar = jnp.array(1e-6, dtype=model_parameters.ar.dtype)
        clipped_ar = jnp.clip(model_parameters.ar, -1.0 + eps_ar, 1.0 - eps_ar)
        return UncARParameters(
            logit_ar=jnp.arctanh(clipped_ar),
            sft_inv_obs_std=jnp.log(jnp.expm1(model_parameters.observation_std)),
            sft_inv_trns_std=jnp.log(jnp.expm1(model_parameters.transition_std)),
        )

    def sample(self, key: PRNGKeyArray) -> UncARParameters:
        ar_key, obs_key, trn_key = jrandom.split(key, 3)

        ar = ar_parameter_sample(ar_key)
        observation_std = obs_std_sample(obs_key)
        transition_std = trns_parameter_sample(trn_key)

        return UncARParameters(
            logit_ar=jnp.arctanh(ar),
            sft_inv_obs_std=jnp.log(jnp.expm1(observation_std)),
            sft_inv_trns_std=jnp.log(jnp.expm1(transition_std)),
        )


    def log_prob(
        self,
        inference_parameters: UncARParameters,
    ) -> Scalar:
        x_ar = inference_parameters.logit_ar
        x_obs = inference_parameters.sft_inv_obs_std
        x_trn = inference_parameters.sft_inv_trns_std

        ar = jnp.tanh(x_ar)
        observation_std = jax.nn.softplus(x_obs)
        transition_std = jax.nn.softplus(x_trn)

        lad_ar = jnp.log1p(-jnp.square(ar))
        lad_obs = jax.nn.log_sigmoid(x_obs)
        lad_trn = jax.nn.log_sigmoid(x_trn)

        return (
            ar_parameter_log_prob(ar)
            + obs_std_log_prob(observation_std)
            + trns_parameter_log_prob(transition_std)
            + lad_ar
            + lad_obs
            + lad_trn
        )
    

@dataclass
class ARBayesian:
    target: typing.ClassVar = ar_model
    parameterization: FullParameterization

def ar_full(hyperparameters: typing.Any) -> ARBayesian:
    return ARBayesian(
        parameterization=FullParameterization()
    )
