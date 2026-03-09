"""
AR(1) model implementations - a univariate LGSSM.
"""
import typing
from dataclasses import field
from typing import ClassVar
from functools import partial
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import (
    LatentContext,
    ObservationContext,
    ConditionContext,
)
from .shared import gaussian_loc_scale_transition

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


class NoisyEmission(Observation):
    """Observation wrapping a scalar value."""

    y: Scalar

    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0

latent_cls = LatentValue
observation_cls = NoisyEmission
parameter_cls = ARParameters
condition_cls = NoCondition

latent_context: typing.Callable[[tuple[LatentValue]], LatentContext[LatentValue]]
latent_context = partial(LatentContext, length=1)
observation_context: typing.Callable[[tuple], ObservationContext[NoisyEmission]] 
observation_context = partial(ObservationContext, length=0)
condition_context: typing.Callable[[tuple], ConditionContext[NoCondition]]
condition_context = partial(ConditionContext, length=0)

def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[NoCondition],
    parameters: ARParameters,
) -> LatentContext[LatentValue]:
    """Sample the initial latent value."""
    stationary_scale = jnp.sqrt(
        jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
    )
    x0 = stationary_scale * jrandom.normal(
        key,
    )
    return latent_context((LatentValue(x=x0),))

def prior_log_prob(
    latent: LatentContext[LatentValue],
    conditions: ConditionContext[NoCondition],
    parameters: ARParameters,
) -> Scalar:
    """Evaluate the prior log-density."""
    stationary_scale = jnp.sqrt(
        jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
    )
    return jstats.norm.logpdf(latent[0].x, scale=stationary_scale)

def ar_loc_scale(
    latent_history: LatentContext[LatentValue],
    condition: NoCondition,
    parameters: ARParameters,
) -> tuple[jax.Array, jax.Array]:
    last_latent = latent_history[0]
    loc_x = parameters.ar * last_latent.x
    scale_x = parameters.transition_std
    return loc_x, scale_x


transition_sample, transition_log_prob = gaussian_loc_scale_transition(
    ar_loc_scale,
    LatentValue,
)

def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[LatentValue],
    observation_history: ObservationContext[NoisyEmission],
    condition: NoCondition,
    parameters: ARParameters,
) -> NoisyEmission:
    """Sample an observation."""
    current_latent = latent_history[0]
    y = current_latent.x + jrandom.normal(key) * parameters.observation_std
    return NoisyEmission(y=y)

def emission_log_prob(
    latent_history: LatentContext[LatentValue],
    observation: NoisyEmission,
    observation_history: ObservationContext[NoisyEmission],
    condition: NoCondition,
    parameters: ARParameters,
) -> Scalar:
    """Return the emission log-density."""
    current_latent = latent_history[0]
    return jstats.norm.logpdf(
        observation.y,
        loc=current_latent.x,
        scale=parameters.observation_std,
    )

