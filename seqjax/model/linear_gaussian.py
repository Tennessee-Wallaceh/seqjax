"""Multidimensional linear Gaussian state-space model on the protocol interface."""

from collections import OrderedDict
from dataclasses import field
from functools import partial
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, PRNGKeyArray, Scalar

from seqjax.model.interface import ConditionContext, LatentContext, ObservationContext
from seqjax.model.typing import Latent, NoCondition, Observation, Parameters


class VectorState50D(Latent):
    """Multivariate latent state."""

    x: Array

    _shape_template: typing.ClassVar = OrderedDict(
        x=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
    )


class VectorObservation50D(Observation):
    """Vector-valued observation."""

    y: Array

    _shape_template: typing.ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
    )


class LGSSMParameters50D(Parameters):
    """Parameters of a 50-dimensional linear Gaussian state-space model."""

    transition_matrix: Array = field(default_factory=lambda: jnp.eye(50))
    transition_noise_scale: Array = field(default_factory=lambda: jnp.ones(50))
    transition_noise_corr_cholesky: Array = field(default_factory=lambda: jnp.eye(50))

    emission_matrix: Array = field(default_factory=lambda: jnp.eye(50))
    emission_noise_scale: Array = field(default_factory=lambda: jnp.ones(50))
    emission_noise_corr_cholesky: Array = field(default_factory=lambda: jnp.eye(50))

    _shape_template: typing.ClassVar = OrderedDict(
        transition_matrix=jax.ShapeDtypeStruct(shape=(50, 50), dtype=jnp.float32),
        transition_noise_scale=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
        transition_noise_corr_cholesky=jax.ShapeDtypeStruct(shape=(50, 50), dtype=jnp.float32),
        emission_matrix=jax.ShapeDtypeStruct(shape=(50, 50), dtype=jnp.float32),
        emission_noise_scale=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
        emission_noise_corr_cholesky=jax.ShapeDtypeStruct(shape=(50, 50), dtype=jnp.float32),
    )

    @property
    def transition_noise_cholesky(self) -> Array:
        return jnp.diag(self.transition_noise_scale) @ self.transition_noise_corr_cholesky

    @property
    def transition_noise_covariance(self) -> Array:
        chol = self.transition_noise_cholesky
        return chol @ chol.T

    @property
    def emission_noise_cholesky(self) -> Array:
        return jnp.diag(self.emission_noise_scale) @ self.emission_noise_corr_cholesky

    @property
    def emission_noise_covariance(self) -> Array:
        chol = self.emission_noise_cholesky
        return chol @ chol.T


LGSSMParameters = LGSSMParameters50D
VectorState = VectorState50D
VectorObservation = VectorObservation50D

prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0

latent_cls = VectorState
observation_cls = VectorObservation
parameter_cls = LGSSMParameters
condition_cls = NoCondition

latent_context: typing.Callable[[tuple[VectorState]], LatentContext[VectorState]]
latent_context = partial(LatentContext, length=1)
observation_context: typing.Callable[[tuple], ObservationContext[VectorObservation]]
observation_context = partial(ObservationContext, length=0)
condition_context: typing.Callable[[tuple], ConditionContext[NoCondition]]
condition_context = partial(ConditionContext, length=0)


def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[NoCondition],
    parameters: LGSSMParameters,
) -> LatentContext[VectorState]:
    """Sample the initial latent state."""
    _ = conditions
    mean = jnp.zeros_like(parameters.transition_noise_scale)
    scale = parameters.transition_noise_scale
    x0 = mean + scale * jrandom.normal(key, shape=scale.shape)
    return latent_context((VectorState(x=x0),))


def prior_log_prob(
    latent: LatentContext[VectorState],
    conditions: ConditionContext[NoCondition],
    parameters: LGSSMParameters,
) -> Scalar:
    """Evaluate the prior log-density."""
    _ = conditions
    scale = parameters.transition_noise_scale
    return jstats.norm.logpdf(latent[0].x, loc=0.0, scale=scale).sum()


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[VectorState],
    condition: NoCondition,
    parameters: LGSSMParameters,
) -> VectorState:
    """Sample the next latent state."""
    _ = condition
    last_state = latent_history[0]
    mean = parameters.transition_matrix @ last_state.x
    noise = parameters.transition_noise_scale * jrandom.normal(
        key,
        shape=parameters.transition_noise_scale.shape,
    )
    return VectorState(x=mean + noise)


def transition_log_prob(
    latent_history: LatentContext[VectorState],
    latent: VectorState,
    condition: NoCondition,
    parameters: LGSSMParameters,
) -> Scalar:
    """Transition log-density under Gaussian innovations."""
    _ = condition
    last_state = latent_history[0]
    mean = parameters.transition_matrix @ last_state.x
    return jstats.norm.logpdf(
        latent.x,
        loc=mean,
        scale=parameters.transition_noise_scale,
    ).sum()


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[VectorState],
    observation_history: ObservationContext[VectorObservation],
    condition: NoCondition,
    parameters: LGSSMParameters,
) -> VectorObservation:
    """Sample an observation from the current latent state."""
    _ = (observation_history, condition)
    state = latent_history[0]
    mean = parameters.emission_matrix @ state.x
    noise = parameters.emission_noise_scale * jrandom.normal(
        key,
        shape=parameters.emission_noise_scale.shape,
    )
    return VectorObservation(y=mean + noise)


def emission_log_prob(
    latent_history: LatentContext[VectorState],
    observation: VectorObservation,
    observation_history: ObservationContext[VectorObservation],
    condition: NoCondition,
    parameters: LGSSMParameters,
) -> Scalar:
    """Emission log-density under Gaussian measurement noise."""
    _ = (observation_history, condition)
    state = latent_history[0]
    mean = parameters.emission_matrix @ state.x
    return jstats.norm.logpdf(
        observation.y,
        loc=mean,
        scale=parameters.emission_noise_scale,
    ).sum()
