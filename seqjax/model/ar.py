"""AR(1) example model implementations."""

from dataclasses import field
from typing import ClassVar
from functools import partial
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    Emission,
    ParameterPrior,
    Prior,
    SequentialModel,
    BayesianSequentialModel,
    GaussianLocScaleTransition,
)
from seqjax.model.typing import (
    HyperParameters,
    Observation,
    NoCondition,
    Parameters,
    Latent,
)


class LatentValue(Latent):
    """Latent AR state."""

    x: Scalar

    _shape_template = OrderedDict(
        x=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class ARParameters(Parameters):
    """Parameters of the AR(1) model."""

    ar: Scalar = field(default_factory=lambda: jnp.array(0.5))
    observation_std: Scalar = field(default_factory=lambda: jnp.array(1.0))
    transition_std: Scalar = field(default_factory=lambda: jnp.array(0.5))

    _shape_template: ClassVar = OrderedDict(
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        observation_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        transition_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class AROnlyParameters(Parameters):
    """Just the AR parameter of the AR(1) model."""

    ar: Scalar = field(default_factory=lambda: jnp.array(0.5))

    _shape_template: ClassVar = OrderedDict(
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class NoisyEmission(Observation):
    """Observation wrapping a scalar value."""

    y: Scalar

    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class HalfCauchyStds(ParameterPrior[ARParameters, HyperParameters]):
    """Half-Cauchy priors for the model standard deviations."""

    @staticmethod
    def sample(key: PRNGKeyArray, _hyperparameters: HyperParameters) -> ARParameters:
        ar_key, o_std_key, t_std_key = jrandom.split(key, 3)
        return ARParameters(
            ar=jrandom.uniform(ar_key, minval=-1, maxval=1),
            observation_std=jnp.abs(jrandom.cauchy(o_std_key)),
            transition_std=jnp.abs(jrandom.cauchy(t_std_key)),
        )

    @staticmethod
    def log_prob(
        parameteters: ARParameters, _hyperparameters: HyperParameters
    ) -> Scalar:
        """Evaluate the log-density of ``parameteters`` under the prior."""
        log_p_theta = jstats.uniform.logpdf(parameteters.ar, loc=-1.0, scale=2.0)
        log_2 = jnp.log(jnp.array(2.0))
        log_p_theta += jstats.cauchy.logpdf(parameteters.observation_std) + log_2
        log_p_theta += jstats.cauchy.logpdf(parameteters.transition_std) + log_2
        return log_p_theta


class AROnlyPrior(ParameterPrior[AROnlyParameters, HyperParameters]):
    @staticmethod
    def sample(
        key: PRNGKeyArray, _hyperparameters: HyperParameters
    ) -> AROnlyParameters:
        return AROnlyParameters(
            ar=jrandom.uniform(key, minval=-1, maxval=1),
        )

    @staticmethod
    def log_prob(
        parameteters: AROnlyParameters, _hyperparameters: HyperParameters
    ) -> Scalar:
        """Evaluate the log-density of ``parameteters`` under the prior."""
        log_p_theta = jstats.uniform.logpdf(parameteters.ar, loc=-1.0, scale=2.0)
        return log_p_theta


def iv_sample(
    key: PRNGKeyArray,
    conditions: tuple[NoCondition],
    parameters: ARParameters,
) -> tuple[LatentValue]:
    """Sample the initial latent value."""
    stationary_scale = jnp.sqrt(
        jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
    )
    x0 = stationary_scale * jrandom.normal(
        key,
    )
    return (LatentValue(x=x0),)


def iv_log_prob(
    latent: tuple[LatentValue],
    conditions: tuple[NoCondition],
    parameters: ARParameters,
) -> Scalar:
    """Evaluate the prior log-density."""
    stationary_scale = jnp.sqrt(
        jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
    )
    return jstats.norm.logpdf(latent[0].x, scale=stationary_scale)


initial_value = Prior[tuple[LatentValue], tuple[NoCondition], ARParameters](
    order=1,
    sample=iv_sample,
    log_prob=iv_log_prob,
)


def ar_loc_scale(
    latent_history: tuple[LatentValue],
    condition: NoCondition,
    parameters: ARParameters,
) -> tuple[jax.Array, jax.Array]:
    (last_latent,) = latent_history
    loc_x = parameters.ar * last_latent.x
    scale_x = parameters.transition_std
    return loc_x, scale_x


ar_random_walk = GaussianLocScaleTransition(
    loc_scale=ar_loc_scale,
    latent_t=LatentValue,
)


def ar_emission_sample(
    key: PRNGKeyArray,
    latent: tuple[LatentValue],
    observation_history: tuple[()],
    condition: NoCondition,
    parameters: ARParameters,
) -> NoisyEmission:
    """Sample an observation."""
    (current_latent,) = latent
    y = current_latent.x + jrandom.normal(key) * parameters.observation_std
    return NoisyEmission(y=y)


def ar_emission_log_prob(
    latent: tuple[LatentValue],
    observation: NoisyEmission,
    observation_history: tuple[()],
    condition: NoCondition,
    parameters: ARParameters,
) -> Scalar:
    """Return the emission log-density."""
    (current_latent,) = latent
    return jstats.norm.logpdf(
        observation.y,
        loc=current_latent.x,
        scale=parameters.observation_std,
    )


ar_emission = Emission[
    tuple[LatentValue],
    NoCondition,
    NoisyEmission,
    ARParameters,
](
    sample=ar_emission_sample,
    log_prob=ar_emission_log_prob,
    order=1,
)


class AR1Target(
    SequentialModel[
        LatentValue,
        NoisyEmission,
        NoCondition,
        ARParameters,
    ]
):
    latent_cls = LatentValue
    observation_cls = NoisyEmission
    parameter_cls = ARParameters
    condition_cls = NoCondition

    prior = initial_value
    transition = ar_random_walk
    emission = ar_emission


def fill_parameter(ar_only: AROnlyParameters, ref_params: ARParameters) -> ARParameters:
    return ARParameters(
        ar_only.ar,
        observation_std=jnp.ones_like(ar_only.ar) * ref_params.observation_std,
        transition_std=jnp.ones_like(ar_only.ar) * ref_params.transition_std,
    )


class AR1Bayesian(
    BayesianSequentialModel[
        LatentValue,
        NoisyEmission,
        NoCondition,
        ARParameters,
        AROnlyParameters,
        HyperParameters,
    ]
):
    def __init__(self, ref_params: ARParameters):
        self.target_parameter = staticmethod(
            partial(fill_parameter, ref_params=ref_params)
        )

    inference_parameter_cls = AROnlyParameters
    target = AR1Target()  # defind for ARParameters
    parameter_prior = AROnlyPrior()  # defined for the partial parameters
