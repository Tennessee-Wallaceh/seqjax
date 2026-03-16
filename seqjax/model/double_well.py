"""Double-well state-space model on the protocol-based model interface."""

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
    ConditionContext,
    LatentContext,
    ObservationContext,
    SequentialModelProtocol,
    validate_sequential_model,
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

latent_context: typing.Callable[[tuple[LatentValue]], LatentContext[LatentValue]]
latent_context = partial(LatentContext, length=1)
observation_context: typing.Callable[[tuple], ObservationContext[NoisyObservation]]
observation_context = partial(ObservationContext, length=0)
condition_context: typing.Callable[[tuple[TimeIncrement]], ConditionContext[TimeIncrement]]
condition_context = partial(ConditionContext, length=1)


def _transition_mean(
    latent_history: LatentContext[LatentValue],
    condition: TimeIncrement,
    parameters: DoubleWellParams,
) -> Scalar:
    previous = latent_history[0].latent_state
    dt = condition.dt
    drift = 4.0 * previous * (jnp.sqrt(parameters.energy_barrier) - previous * previous)
    return previous + dt * drift


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DoubleWellModel(
    SequentialModelProtocol[
        LatentValue,
        NoisyObservation,
        TimeIncrement,
        DoubleWellParams,
    ]
):
    prior_order: int = 1
    transition_order: int = 1
    emission_order: int = 1
    observation_dependency: int = 0

    latent_cls: type[LatentValue] = LatentValue
    observation_cls: type[NoisyObservation] = NoisyObservation
    parameter_cls: type[DoubleWellParams] = DoubleWellParams
    condition_cls: type[TimeIncrement] = TimeIncrement

    latent_context: typing.Callable[..., LatentContext[LatentValue]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[NoisyObservation]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[TimeIncrement]] = condition_context

    @staticmethod
    def prior_sample(
        key: PRNGKeyArray,
        conditions: ConditionContext[TimeIncrement],
        parameters: DoubleWellParams,
    ) -> LatentContext[LatentValue]:
        """Sample the initial latent value from a unit Gaussian."""
        _ = (conditions, parameters)
        x0 = jrandom.normal(key)
        return latent_context((LatentValue(latent_state=x0),))

    @staticmethod
    def prior_log_prob(
        latent: LatentContext[LatentValue],
        conditions: ConditionContext[TimeIncrement],
        parameters: DoubleWellParams,
    ) -> Scalar:
        """Evaluate the prior log-density for the initial latent."""
        _ = (conditions, parameters)
        return jstats.norm.logpdf(latent[0].latent_state, scale=jnp.array(1.0))

    @staticmethod
    def transition_sample(
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentValue],
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> LatentValue:
        """Sample next latent by Euler-Maruyama discretisation."""
        mean = _transition_mean(latent_history, condition, parameters)
        scale = parameters.transition_std * jnp.sqrt(condition.dt)
        return LatentValue(latent_state=mean + jrandom.normal(key) * scale)

    @staticmethod
    def transition_log_prob(
        latent_history: LatentContext[LatentValue],
        latent: LatentValue,
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> Scalar:
        """Transition log-density under Gaussian discretisation noise."""
        mean = _transition_mean(latent_history, condition, parameters)
        scale = parameters.transition_std * jnp.sqrt(condition.dt)
        return jstats.norm.logpdf(latent.latent_state, loc=mean, scale=scale)

    @staticmethod
    def emission_sample(
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentValue],
        observation_history: ObservationContext[NoisyObservation],
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> NoisyObservation:
        """Sample noisy observation from the current latent state."""
        _ = (observation_history, condition)
        current = latent_history[0].latent_state
        y = current + jrandom.normal(key) * parameters.observation_std
        return NoisyObservation(observation=y)

    @staticmethod
    def emission_log_prob(
        latent_history: LatentContext[LatentValue],
        observation: NoisyObservation,
        observation_history: ObservationContext[NoisyObservation],
        condition: TimeIncrement,
        parameters: DoubleWellParams,
    ) -> Scalar:
        """Observation log-density given current latent state."""
        _ = (observation_history, condition)
        current = latent_history[0].latent_state
        return jstats.norm.logpdf(
            observation.observation,
            loc=current,
            scale=parameters.observation_std,
        )


double_well_model = validate_sequential_model(DoubleWellModel())


def make_unit_time_increments(
    sequence_length: int,
    *,
    dt: float = 1.0,
) -> TimeIncrement:
    """Return a ``TimeIncrement`` tree filled with constant ``dt``."""
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    required_length = sequence_length + double_well_model.prior_order - 1
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
