"""Double-well state-space model on the protocol-based model interface."""

from collections import OrderedDict
from dataclasses import field
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import (
    LatentContext,
    ObservedHistoryContext,
    SequentialModel,
)
from seqjax.model.typing import (
    Observation,
    Parameters,
    Condition,
    Latent,
)


class LatentValue(Latent):
    """Latent state for the double-well process."""

    latent_state: Scalar

    _shape_template = OrderedDict(
        latent_state=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class DoubleWellParams(Parameters):
    """Full parameter set of the double-well model."""

    energy_barrier: Scalar = field(default_factory=lambda: jnp.array(0.5))
    observation_std: Scalar = field(default_factory=lambda: jnp.array(1.0))
    transition_std: Scalar = field(default_factory=lambda: jnp.array(0.5))

    _shape_template: typing.ClassVar = OrderedDict(
        energy_barrier=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        observation_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        transition_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class EBOnlyParameters(Parameters):
    """Inference parameterisation with only the energy-barrier free."""

    energy_barrier: Scalar = field(default_factory=lambda: jnp.array(0.5))

    _shape_template: typing.ClassVar = OrderedDict(
        energy_barrier=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class TimeIncrement(Condition):
    """Time-step condition between observations."""

    dt: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        dt=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class NoisyObservation(Observation):
    """Observed scalar with Gaussian noise."""

    observation: Scalar

    _shape_template: typing.ClassVar = OrderedDict(
        observation=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


latent_cls = LatentValue
observation_cls = NoisyObservation
parameter_cls = DoubleWellParams
condition_cls = TimeIncrement

def _transition_mean(
    latent_history: LatentContext[LatentValue, typing.Literal[1]],
    condition: TimeIncrement,
    parameters: DoubleWellParams,
) -> Scalar:
    previous = latent_history[-1].latent_state
    dt = condition.dt
    drift = 4.0 * previous * (jnp.sqrt(parameters.energy_barrier) - previous * previous)
    return previous + dt * drift


def prior_sample(
    key: PRNGKeyArray,
    parameters: DoubleWellParams,
) -> LatentContext[LatentValue, typing.Literal[1]]:
    """Sample the initial latent value from a unit Gaussian."""
    _ = parameters
    x0 = jrandom.normal(key)
    return LatentContext.from_values(LatentValue(latent_state=x0), length=1)


def prior_log_prob(
    latent: LatentContext[LatentValue, typing.Literal[1]],
    parameters: DoubleWellParams,
) -> Scalar:
    """Evaluate the prior log-density for the initial latent."""
    _ = parameters
    return jstats.norm.logpdf(latent[-1].latent_state, scale=jnp.array(1.0))


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentValue, typing.Literal[1]],
    parameters: DoubleWellParams,
    condition: TimeIncrement,
    observation_history: ObservedHistoryContext[NoisyObservation, TimeIncrement, typing.Literal[0]],
) -> LatentValue:
    _ = observation_history
    """Sample next latent by Euler-Maruyama discretisation."""
    mean = _transition_mean(latent_history, condition, parameters)
    scale = parameters.transition_std * jnp.sqrt(condition.dt)
    return LatentValue(latent_state=mean + jrandom.normal(key) * scale)


def transition_log_prob(
    latent: LatentValue,
    latent_history: LatentContext[LatentValue, typing.Literal[1]],
    parameters: DoubleWellParams,
    condition: TimeIncrement,
    observation_history: ObservedHistoryContext[NoisyObservation, TimeIncrement, typing.Literal[0]],
) -> Scalar:
    _ = observation_history
    """Transition log-density under Gaussian discretisation noise."""
    mean = _transition_mean(latent_history, condition, parameters)
    scale = parameters.transition_std * jnp.sqrt(condition.dt)
    return jstats.norm.logpdf(latent.latent_state, loc=mean, scale=scale)


def emission_sample(
    key: PRNGKeyArray,
    current_latent: LatentValue,
    parameters: DoubleWellParams,
    condition: TimeIncrement,
    latent_history: LatentContext[LatentValue, typing.Literal[1]],
    observation_history: ObservedHistoryContext[NoisyObservation, TimeIncrement, typing.Literal[0]],
) -> NoisyObservation:
    """Sample noisy observation from the current latent state."""
    _ = (observation_history, condition)
    y = current_latent.latent_state + jrandom.normal(key) * parameters.observation_std
    return NoisyObservation(observation=y)


def emission_log_prob(
    observation: NoisyObservation,
    current_latent: LatentValue,
    parameters: DoubleWellParams,
    condition: TimeIncrement,
    latent_history: LatentContext[LatentValue, typing.Literal[1]],
    observation_history: ObservedHistoryContext[NoisyObservation, TimeIncrement, typing.Literal[0]],
) -> Scalar:
    """Observation log-density given current latent state."""
    _ = (observation_history, condition)
    return jstats.norm.logpdf(
        observation.observation,
        loc=current_latent.latent_state,
        scale=parameters.observation_std,
    )




double_well_model = SequentialModel(
    latent_cls=latent_cls, observation_cls=observation_cls,
    parameter_cls=parameter_cls, condition_cls=condition_cls,
    transition_latent_order=1, transition_observation_order=0,
    emission_latent_order=0, emission_observation_order=0,
    prior_sample=prior_sample, prior_log_prob=prior_log_prob,
    transition_sample=transition_sample, transition_log_prob=transition_log_prob,
    emission_sample=emission_sample, emission_log_prob=emission_log_prob,
)


def make_unit_time_increments(
    sequence_length: int,
    *,
    dt: float = 1.0,
) -> TimeIncrement:
    """Return a ``TimeIncrement`` tree filled with constant ``dt``."""
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    required_length = sequence_length
    dt_value = jnp.asarray(dt, dtype=jnp.float32)
    return TimeIncrement(dt=jnp.full((required_length,), dt_value, dtype=dt_value.dtype))


def fill_parameter(
    eb_only: EBOnlyParameters,
    ref_params: DoubleWellParams,
) -> DoubleWellParams:
    """Lift the reduced EB-only parameters into full model parameters."""
    return DoubleWellParams(
        energy_barrier=eb_only.energy_barrier,
        observation_std=jnp.ones_like(eb_only.energy_barrier) * ref_params.observation_std,
        transition_std=jnp.ones_like(eb_only.energy_barrier) * ref_params.transition_std,
    )
