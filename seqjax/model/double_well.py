"""Double Well model implementations."""

from dataclasses import field
from typing import ClassVar
from functools import partial
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats

import jaxtyping

import seqjax.model.typing as seqjtyping
from seqjax.model.base import (
    Emission,
    ParameterPrior,
    Prior,
    SequentialModel,
    BayesianSequentialModel,
    Transition,
)


class LatentValue(seqjtyping.Particle):
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

    dt: jaxtyping.Scalar  # time since last observation

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


class InitialValue(Prior[tuple[LatentValue], tuple[()], DoubleWellParams]):
    """Gaussian prior over the initial latent state."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: jaxtyping.PRNGKeyArray,
        conditions: tuple[()],
        parameters: DoubleWellParams,
    ) -> tuple[LatentValue]:
        """Sample the initial latent value."""
        scale = jnp.array(1.0)
        x0 = scale * jrandom.normal(
            key,
        )
        return (LatentValue(latent_state=x0),)

    @staticmethod
    def log_prob(
        particle: tuple[LatentValue],
        conditions: tuple[()],
        parameters: DoubleWellParams,
    ) -> jaxtyping.Scalar:
        """Evaluate the prior log-density."""
        scale = jnp.array(1.0)
        return jstats.norm.logpdf(particle[0].latent_state, scale=scale)


class DoubleWellWalk(
    Transition[LatentValue, tuple[LatentValue], TimeIncrement, DoubleWellParams]
):
    order: ClassVar[int] = 1

    @staticmethod
    def mean_fn(
        last_particle: LatentValue,
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> jaxtyping.Scalar:
        last_latent = last_particle.latent_state
        dt = condition.dt
        return last_latent + dt * 4 * last_latent * (
            jnp.sqrt(parameters.energy_barrier) - last_latent * last_latent
        )

    @staticmethod
    def sample(
        key: jaxtyping.PRNGKeyArray,
        particle_history: tuple[LatentValue],
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> LatentValue:
        (last_particle,) = particle_history
        mean = DoubleWellWalk.mean_fn(last_particle, condition, parameters)
        dt = condition.dt
        return LatentValue(
            latent_state=mean
            + jrandom.normal(key) * parameters.transition_std * jnp.sqrt(dt)
        )

    @staticmethod
    def log_prob(
        particle_history: tuple[LatentValue],
        particle: LatentValue,
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> jaxtyping.Scalar:
        (last_particle,) = particle_history
        mean = DoubleWellWalk.mean_fn(last_particle, condition, parameters)
        dt = condition.dt
        return jstats.norm.logpdf(
            particle.latent_state,
            loc=mean,
            scale=parameters.transition_std * jnp.sqrt(dt),
        )


class NoisyEmission(
    Emission[
        tuple[LatentValue], NoisyObservation, tuple[()], TimeIncrement, DoubleWellParams
    ]
):
    """Normal emission from the latent state."""

    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(  # type: ignore[override]
        key: jaxtyping.PRNGKeyArray,
        particle: tuple[LatentValue],
        observation_history: tuple[()],
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> NoisyObservation:
        """Sample an observation."""
        (current_particle,) = particle
        y = (
            current_particle.latent_state
            + jrandom.normal(key) * parameters.observation_std
        )
        return NoisyObservation(observation=y)

    @staticmethod
    def log_prob(  # type: ignore[override]
        particle: tuple[LatentValue],
        observation_history: tuple[()],
        observation: NoisyObservation,
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> jaxtyping.Scalar:
        """Return the emission log-density."""
        (current_particle,) = particle
        return jstats.norm.logpdf(
            observation.observation,
            loc=current_particle.latent_state,
            scale=parameters.observation_std,
        )


class DoubleWellTarget(
    SequentialModel[
        LatentValue,  # particle
        tuple[LatentValue],  # initial particles
        tuple[LatentValue],  # transtiion history
        tuple[LatentValue],  # observation history
        NoisyObservation,  # observation
        tuple[()],  # observation history
        tuple[()],  # prior conditions
        TimeIncrement,  # condition
        DoubleWellParams,  # parameters
    ]
):
    particle_cls = LatentValue
    observation_cls = NoisyObservation
    parameter_cls = DoubleWellParams
    condition_cls = TimeIncrement
    prior = InitialValue()
    transition = DoubleWellWalk()
    emission = NoisyEmission()


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

    required_length = sequence_length + DoubleWellTarget.prior.order - 1
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
        LatentValue,  # particle
        tuple[LatentValue],  # initial particles
        tuple[LatentValue],  # transtiion history
        tuple[LatentValue],  # observation history
        NoisyObservation,  # observation
        tuple[()],  # observation history
        tuple[()],  # prior conditions
        TimeIncrement,  # condition
        DoubleWellParams,  # parameters
        EBOnlyParameters,
        seqjtyping.HyperParameters,
    ]
):
    def __init__(self, ref_params: DoubleWellParams):
        self.target_parameter = staticmethod(
            partial(fill_parameter, ref_params=ref_params)
        )

    inference_parameter_cls = EBOnlyParameters
    target = DoubleWellTarget()  # defind for full parameters
    parameter_prior = EBOnlyPrior()  # defined for the partial parameters
