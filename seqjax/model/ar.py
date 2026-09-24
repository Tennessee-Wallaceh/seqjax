"""
AR(1) model implementations - a univariate LGSSM.
"""
from dataclasses import field
from typing import ClassVar
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import (
    SequentialModel,
    LatentContext,
    ObservedHistoryContext,
)

from seqjax.model.typing import (
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


class NoisyEmission(Observation):
    """Observation wrapping a scalar value."""

    y: Scalar

    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


latent_cls = LatentValue
observation_cls = NoisyEmission
parameter_cls = ARParameters
condition_cls = NoCondition

def _ar_loc_scale(
    latent_history: LatentContext[LatentValue],
    condition: NoCondition,
    parameters: ARParameters,
) -> tuple[jax.Array, jax.Array]:
    del condition
    last_latent = latent_history[-1]
    loc_x = parameters.ar * last_latent.x
    scale_x = parameters.transition_std
    return loc_x, scale_x


def prior_sample(
    key: PRNGKeyArray,
    parameters: ARParameters,
) -> LatentContext[LatentValue]:
    stationary_scale = jnp.sqrt(
        jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
    )
    x0 = stationary_scale * jrandom.normal(key)
    return LatentContext.from_values(LatentValue(x=x0), length=1)


def prior_log_prob(
    latent: LatentContext[LatentValue],
    parameters: ARParameters,
) -> Scalar:
    stationary_scale = jnp.sqrt(
        jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
    )
    return jstats.norm.logpdf(latent[-1].x, scale=stationary_scale)


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentValue],
    parameters: ARParameters,
    condition: NoCondition,
    observation_history: ObservedHistoryContext[NoisyEmission, NoCondition],
) -> LatentValue:
    _ = observation_history
    loc_x, scale_x = _ar_loc_scale(latent_history, condition, parameters)
    eps = jrandom.normal(key)
    next_x = loc_x + eps * scale_x
    return LatentValue.unravel(next_x)


def transition_log_prob(
    latent_history: LatentContext[LatentValue],
    latent: LatentValue,
    parameters: ARParameters,
    condition: NoCondition,
    observation_history: ObservedHistoryContext[NoisyEmission, NoCondition],
) -> Scalar:
    _ = observation_history
    loc_x, scale_x = _ar_loc_scale(latent_history, condition, parameters)
    x = latent.ravel()
    lp = jstats.norm.logpdf(x, loc=loc_x, scale=scale_x)
    return jnp.sum(lp)


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentValue],
    parameters: ARParameters,
    condition: NoCondition,
    observation_history: ObservedHistoryContext[NoisyEmission, NoCondition],
) -> NoisyEmission:
    del observation_history
    del condition
    current_latent = latent_history[-1]
    y = current_latent.x + jrandom.normal(key) * parameters.observation_std
    return NoisyEmission(y=y)


def emission_log_prob(
    latent_history: LatentContext[LatentValue],
    observation: NoisyEmission,
    parameters: ARParameters,
    condition: NoCondition,
    observation_history: ObservedHistoryContext[NoisyEmission, NoCondition],
) -> Scalar:
    del observation_history
    del condition
    current_latent = latent_history[-1]
    return jstats.norm.logpdf(
        observation.y,
        loc=current_latent.x,
        scale=parameters.observation_std,
    )




ar_model = SequentialModel(
    latent_cls=latent_cls, observation_cls=observation_cls,
    parameter_cls=parameter_cls, condition_cls=condition_cls,
    transition_latent_order=1, transition_observation_order=0,
    emission_latent_order=1, emission_observation_order=0,
    prior_sample=prior_sample, prior_log_prob=prior_log_prob,
    transition_sample=transition_sample, transition_log_prob=transition_log_prob,
    emission_sample=emission_sample, emission_log_prob=emission_log_prob,
)