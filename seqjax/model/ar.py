"""
AR(1) model implementations - a univariate LGSSM.
"""
import typing
from dataclasses import field, dataclass
from typing import ClassVar
from collections import OrderedDict
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.interface import (
    SequentialModelProtocol,
    validate_sequential_model,
    LatentContext,
    ObservationContext,
    ConditionContext,
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

latent_context: typing.Callable[[tuple[LatentValue]], LatentContext[LatentValue]]
latent_context = partial(LatentContext, length=1)
observation_context: typing.Callable[[tuple], ObservationContext[NoisyEmission]] 
observation_context: typing.Callable[[tuple], ObservationContext[NoisyEmission]]
observation_context = partial(ObservationContext, length=0)
condition_context: typing.Callable[[tuple], ConditionContext[NoCondition]]
condition_context = partial(ConditionContext, length=0)

@jax.tree_util.register_dataclass
@dataclass
class ARModel(
    SequentialModelProtocol[
        LatentValue,
        NoisyEmission,
        NoCondition,
        ARParameters,
    ]
):
    latent_cls: type[LatentValue] = LatentValue
    observation_cls: type[NoisyEmission] = NoisyEmission
    parameter_cls: type[ARParameters] = ARParameters
    condition_cls: type[NoCondition] = NoCondition

    prior_order: int = 1
    transition_order: int = 1
    emission_order: int = 1
    observation_dependency: int = 0

    latent_context: typing.Callable[..., LatentContext[LatentValue]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[NoisyEmission]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[NoCondition]] = condition_context

    @staticmethod
    def _ar_loc_scale(
        latent_history: LatentContext[LatentValue],
        condition: NoCondition,
        parameters: ARParameters,
    ) -> tuple[jax.Array, jax.Array]:
        del condition
        last_latent = latent_history[0]
        loc_x = parameters.ar * last_latent.x
        scale_x = parameters.transition_std
        return loc_x, scale_x

    @staticmethod
    def prior_sample(
        key: PRNGKeyArray,
        conditions: ConditionContext[NoCondition],
        parameters: ARParameters,
    ) -> LatentContext[LatentValue]:
        del conditions
        stationary_scale = jnp.sqrt(
            jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
        )
        x0 = stationary_scale * jrandom.normal(key)
        return latent_context((LatentValue(x=x0),))

    @staticmethod
    def prior_log_prob(
        latent: LatentContext[LatentValue],
        conditions: ConditionContext[NoCondition],
        parameters: ARParameters,
    ) -> Scalar:
        del conditions
        stationary_scale = jnp.sqrt(
            jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
        )
        return jstats.norm.logpdf(latent[0].x, scale=stationary_scale)

    @staticmethod
    def transition_sample(
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentValue],
        condition: NoCondition,
        parameters: ARParameters,
    ) -> LatentValue:
        loc_x, scale_x = ARModel._ar_loc_scale(latent_history, condition, parameters)
        eps = jrandom.normal(key)
        next_x = loc_x + eps * scale_x
        return LatentValue.unravel(next_x)

    @staticmethod
    def transition_log_prob(
        latent_history: LatentContext[LatentValue],
        latent: LatentValue,
        condition: NoCondition,
        parameters: ARParameters,
    ) -> Scalar:
        loc_x, scale_x = ARModel._ar_loc_scale(latent_history, condition, parameters)
        x = latent.ravel()
        lp = jstats.norm.logpdf(x, loc=loc_x, scale=scale_x)
        return jnp.sum(lp)

    @staticmethod
    def emission_sample(
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentValue],
        observation_history: ObservationContext[NoisyEmission],
        condition: NoCondition,
        parameters: ARParameters,
    ) -> NoisyEmission:
        del observation_history
        del condition
        current_latent = latent_history[0]
        y = current_latent.x + jrandom.normal(key) * parameters.observation_std
        return NoisyEmission(y=y)

    @staticmethod
    def emission_log_prob(
        latent_history: LatentContext[LatentValue],
        observation: NoisyEmission,
        observation_history: ObservationContext[NoisyEmission],
        condition: NoCondition,
        parameters: ARParameters,
    ) -> Scalar:
        del observation_history
        del condition
        current_latent = latent_history[0]
        return jstats.norm.logpdf(
            observation.y,
            loc=current_latent.x,
            scale=parameters.observation_std,
        )


ar_model = validate_sequential_model(ARModel())