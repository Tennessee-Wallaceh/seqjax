"""AR(1) example model implementations."""

from dataclasses import field
from typing import ClassVar
from functools import partial
from collections import OrderedDict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    ParameterPrior,
    BayesianSequentialModel,
    SequentialModelBase,
    LatentContext,
    ObservationContext,
    ConditionContext,
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


class AR1Target(
    SequentialModelBase[
        LatentValue,
        NoisyEmission,
        NoCondition,
        ARParameters,
    ]
):
    """Method-based AR(1) target model."""

    latent_cls = LatentValue
    observation_cls = NoisyEmission
    parameter_cls = ARParameters
    condition_cls = NoCondition

    prior_order = 1
    transition_order = 1
    emission_order = 1
    observation_dependency = 0

    prior = SimpleNamespace(order=prior_order)
    transition = SimpleNamespace(order=transition_order)
    emission = SimpleNamespace(
        order=emission_order,
        observation_dependency=observation_dependency,
    )

    @staticmethod
    def _stationary_scale(parameters: ARParameters) -> Scalar:
        return jnp.sqrt(
            jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
        )

    @staticmethod
    def _transition_loc_scale(
        previous_latent: LatentValue,
        parameters: ARParameters,
    ) -> tuple[Scalar, Scalar]:
        loc_x = parameters.ar * previous_latent.x
        scale_x = parameters.transition_std
        return loc_x, scale_x

    @staticmethod
    def _emission_loc_scale(
        current_latent: LatentValue,
        parameters: ARParameters,
    ) -> tuple[Scalar, Scalar]:
        return current_latent.x, parameters.observation_std

    def prior_sample(
        self,
        key: PRNGKeyArray,
        conditions: ConditionContext[NoCondition],
        parameters: ARParameters,
    ) -> LatentContext[LatentValue]:
        del conditions
        stationary_scale = self._stationary_scale(parameters)
        x0 = stationary_scale * jrandom.normal(key)
        return self.make_latent_context(LatentValue(x=x0))

    def prior_log_prob(
        self,
        latent: LatentContext[LatentValue],
        conditions: ConditionContext[NoCondition],
        parameters: ARParameters,
    ) -> Scalar:
        del conditions
        stationary_scale = self._stationary_scale(parameters)
        return jstats.norm.logpdf(latent[-1].x, scale=stationary_scale)

    def transition_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentValue],
        condition: NoCondition,
        parameters: ARParameters,
    ) -> LatentValue:
        del condition
        loc_x, scale_x = self._transition_loc_scale(latent_history[-1], parameters)
        x_next = loc_x + jrandom.normal(key) * scale_x
        return LatentValue(x=x_next)

    def transition_log_prob(
        self,
        latent_history: LatentContext[LatentValue],
        latent: LatentValue,
        condition: NoCondition,
        parameters: ARParameters,
    ) -> Scalar:
        del condition
        loc_x, scale_x = self._transition_loc_scale(latent_history[-1], parameters)
        return jstats.norm.logpdf(latent.x, loc=loc_x, scale=scale_x)

    def emission_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentValue],
        observation_history: ObservationContext[NoisyEmission],
        condition: NoCondition,
        parameters: ARParameters,
    ) -> NoisyEmission:
        del observation_history, condition
        loc_y, scale_y = self._emission_loc_scale(latent_history[-1], parameters)
        y = loc_y + jrandom.normal(key) * scale_y
        return NoisyEmission(y=y)

    def emission_log_prob(
        self,
        latent_history: LatentContext[LatentValue],
        observation: NoisyEmission,
        observation_history: ObservationContext[NoisyEmission],
        condition: NoCondition,
        parameters: ARParameters,
    ) -> Scalar:
        del observation_history, condition
        loc_y, scale_y = self._emission_loc_scale(latent_history[-1], parameters)
        return jstats.norm.logpdf(observation.y, loc=loc_y, scale=scale_y)


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
    def __init__(self, ref_params: ARParameters, hyperparameters: HyperParameters | None = None):
        super().__init__(hyperparameters=hyperparameters)
        self.convert_to_model_parameters = staticmethod(
            partial(fill_parameter, ref_params=ref_params)
        )

    inference_parameter_cls = AROnlyParameters
    target = AR1Target()
    def parameter_prior(self) -> AROnlyPrior:
        return AROnlyPrior()
