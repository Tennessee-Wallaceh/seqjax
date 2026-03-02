"""Double Well model implementations."""

from dataclasses import field
from typing import ClassVar
from functools import partial
from collections import OrderedDict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats

import jaxtyping

import seqjax.model.typing as seqjtyping
from seqjax.model.base import (
    ParameterPrior,
    BayesianSequentialModel,
    SequentialModelBase,
    LatentContext,
    ObservationContext,
    ConditionContext,
)


class LatentValue(seqjtyping.Latent):
    """Latent AR state."""

    latent_state: jaxtyping.Scalar

    _shape_template: ClassVar = OrderedDict(
        latent_state=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class DoubleWellParams(seqjtyping.Parameters):
    """Parameters of the Double Well model."""

    energy_barrier: jaxtyping.Scalar = field(default_factory=lambda: jnp.array(0.5))
    observation_std: jaxtyping.Scalar = field(default_factory=lambda: jnp.array(1.0))
    transition_std: jaxtyping.Scalar = field(default_factory=lambda: jnp.array(0.5))

    _shape_template: ClassVar = OrderedDict(
        energy_barrier=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        observation_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        transition_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class EBOnlyParameters(seqjtyping.Parameters):
    """Just the Energy Barrier parameter of the Double Well model."""

    energy_barrier: jaxtyping.Scalar = field(default_factory=lambda: jnp.array(0.5))

    _shape_template: ClassVar = OrderedDict(
        energy_barrier=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class TimeIncrement(seqjtyping.Condition):
    """Time step between observations."""

    dt: jaxtyping.Scalar

    _shape_template: ClassVar = OrderedDict(
        dt=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class NoisyObservation(seqjtyping.Observation):
    """Observation wrapping a scalar value."""

    observation: jaxtyping.Scalar

    _shape_template: ClassVar = OrderedDict(
        observation=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class EBOnlyPrior(ParameterPrior[EBOnlyParameters, seqjtyping.HyperParameters]):
    @staticmethod
    def sample(
        key: jaxtyping.PRNGKeyArray, _hyperparameters: seqjtyping.HyperParameters
    ) -> EBOnlyParameters:
        return EBOnlyParameters(
            energy_barrier=jrandom.lognormal(key, sigma=jnp.array(1.0)),
        )

    @staticmethod
    def log_prob(
        parameteters: EBOnlyParameters, _hyperparameters: seqjtyping.HyperParameters
    ) -> jaxtyping.Scalar:
        """Evaluate the log-density of ``parameteters`` under the prior."""
        log_p_theta = jstats.norm.logpdf(
            jnp.log(parameteters.energy_barrier), scale=jnp.array(1.0)
        )
        return log_p_theta


class DoubleWellTarget(
    SequentialModelBase[
        LatentValue,
        NoisyObservation,
        TimeIncrement,
        DoubleWellParams,
    ]
):
    latent_cls = LatentValue
    observation_cls = NoisyObservation
    parameter_cls = DoubleWellParams
    condition_cls = TimeIncrement

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
    def _transition_mean(
        last_latent: LatentValue,
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> jaxtyping.Scalar:
        last_latent_state = last_latent.latent_state
        dt = condition.dt
        return last_latent_state + dt * 4 * last_latent_state * (
            jnp.sqrt(parameters.energy_barrier) - last_latent_state * last_latent_state
        )

    def prior_sample(
        self,
        key: jaxtyping.PRNGKeyArray,
        conditions: ConditionContext[TimeIncrement],
        parameters: DoubleWellParams,
    ) -> LatentContext[LatentValue]:
        del conditions, parameters
        x0 = jrandom.normal(key)
        return self.make_latent_context(LatentValue(latent_state=x0))

    def prior_log_prob(
        self,
        latent: LatentContext[LatentValue],
        conditions: ConditionContext[TimeIncrement],
        parameters: DoubleWellParams,
    ) -> jaxtyping.Scalar:
        del conditions, parameters
        return jstats.norm.logpdf(latent[-1].latent_state, scale=jnp.array(1.0))

    def transition_sample(
        self,
        key: jaxtyping.PRNGKeyArray,
        latent_history: LatentContext[LatentValue],
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> LatentValue:
        mean = self._transition_mean(latent_history[-1], condition, parameters)
        dt = condition.dt
        return LatentValue(
            latent_state=mean
            + jrandom.normal(key) * parameters.transition_std * jnp.sqrt(dt)
        )

    def transition_log_prob(
        self,
        latent_history: LatentContext[LatentValue],
        latent: LatentValue,
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> jaxtyping.Scalar:
        mean = self._transition_mean(latent_history[-1], condition, parameters)
        dt = condition.dt
        return jstats.norm.logpdf(
            latent.latent_state,
            loc=mean,
            scale=parameters.transition_std * jnp.sqrt(dt),
        )

    def emission_sample(
        self,
        key: jaxtyping.PRNGKeyArray,
        latent_history: LatentContext[LatentValue],
        observation_history: ObservationContext[NoisyObservation],
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> NoisyObservation:
        del observation_history, condition
        y = (
            latent_history[-1].latent_state
            + jrandom.normal(key) * parameters.observation_std
        )
        return NoisyObservation(observation=y)

    def emission_log_prob(
        self,
        latent_history: LatentContext[LatentValue],
        observation: NoisyObservation,
        observation_history: ObservationContext[NoisyObservation],
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> jaxtyping.Scalar:
        del observation_history, condition
        return jstats.norm.logpdf(
            observation.observation,
            loc=latent_history[-1].latent_state,
            scale=parameters.observation_std,
        )


def make_unit_time_increments(
    sequence_length: int,
    *,
    dt: float = 1.0,
) -> TimeIncrement:
    """Return a ``TimeIncrement`` tree filled with a constant ``dt``."""

    if sequence_length < 1:
        raise ValueError(
            f"sequence_length must be >= 1, got {sequence_length}",
        )

    required_length = sequence_length + DoubleWellTarget.prior_order - 1
    dt_value = jnp.asarray(dt, dtype=jnp.float32)
    increments = jnp.full((required_length,), dt_value, dtype=dt_value.dtype)
    return TimeIncrement(dt=increments)


def fill_parameter(
    eb_only: EBOnlyParameters, ref_params: DoubleWellParams
) -> DoubleWellParams:
    return DoubleWellParams(
        eb_only.energy_barrier,
        observation_std=jnp.ones_like(eb_only.energy_barrier)
        * ref_params.observation_std,
        transition_std=jnp.ones_like(eb_only.energy_barrier)
        * ref_params.transition_std,
    )


class DoubleWellBayesian(
    BayesianSequentialModel[
        LatentValue,
        NoisyObservation,
        TimeIncrement,
        DoubleWellParams,
        EBOnlyParameters,
        seqjtyping.HyperParameters,
    ]
):
    def __init__(
        self,
        ref_params: DoubleWellParams,
        hyperparameters: seqjtyping.HyperParameters | None = None,
    ):
        super().__init__(hyperparameters=hyperparameters)
        self.convert_to_model_parameters = staticmethod(
            partial(fill_parameter, ref_params=ref_params)
        )

    inference_parameter_cls = EBOnlyParameters
    target = DoubleWellTarget()
    def parameter_prior(self) -> EBOnlyPrior:
        return EBOnlyPrior()
