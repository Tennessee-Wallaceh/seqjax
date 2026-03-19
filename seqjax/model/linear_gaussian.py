"""Multidimensional linear Gaussian state-space model on the protocol interface."""

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, PRNGKeyArray, Scalar
import jax.scipy as jsp

from seqjax.model.interface import (
    ConditionContext,
    LatentContext,
    ObservationContext,
    SequentialModelProtocol,
    validate_sequential_model,
)
from seqjax.model.typing import Latent, NoCondition, Observation, Parameters

"""
Multivariate normal helpers, defined for chol
"""
def _mvn_sample(
    key: PRNGKeyArray,
    mean: Array,
    chol: Array,
) -> Array:
    """Sample from N(mean, chol @ chol.T)."""
    z = jrandom.normal(key, shape=mean.shape, dtype=mean.dtype)
    return mean + chol @ z


def _mvn_log_prob(
    x: Array,
    mean: Array,
    chol: Array,
) -> Scalar:
    """Log-density of N(mean, chol @ chol.T), with lower-triangular chol."""
    diff = x - mean
    whitened = jsp.linalg.solve_triangular(
        chol,
        diff,
        lower=True,
    )
    log_det_chol = jnp.sum(jnp.log(jnp.diag(chol)))
    dim = diff.shape[0]
    return (
        -0.5 * jnp.dot(whitened, whitened)
        - log_det_chol
        - 0.5 * dim * jnp.log(2.0 * jnp.pi)
    )


class VectorState5D(Latent):
    """Multivariate latent state."""

    x: Array

    _shape_template: typing.ClassVar = OrderedDict(
        x=jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32),
    )


class VectorObservation5D(Observation):
    """Vector-valued observation."""

    y: Array

    _shape_template: typing.ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32),
    )


class LGSSMParameters5D(Parameters):
    """Canonical 5D LGSSM parameters."""

    transition_matrix: Array = field(default_factory=lambda: 0.7 * jnp.eye(5))
    transition_noise_cholesky: Array = field(default_factory=lambda: jnp.eye(5))

    emission_matrix: Array = field(default_factory=lambda: jnp.eye(5))
    emission_noise_cholesky: Array = field(default_factory=lambda: jnp.eye(5))

    _shape_template: typing.ClassVar = OrderedDict(
        transition_matrix=jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
        transition_noise_cholesky=jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
        emission_matrix=jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
        emission_noise_cholesky=jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
    )

    @property
    def transition_noise_covariance(self) -> Array:
        L = self.transition_noise_cholesky
        return L @ jnp.swapaxes(L, -1, -2)

    @property
    def emission_noise_covariance(self) -> Array:
        L = self.emission_noise_cholesky
        return L @ jnp.swapaxes(L, -1, -2)


LGSSMParameters = LGSSMParameters5D
VectorState = VectorState5D
VectorObservation = VectorObservation5D

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
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LGSSMModel(
    SequentialModelProtocol[
        VectorState,
        VectorObservation,
        NoCondition,
        LGSSMParameters,
    ]
):
    prior_order: int = 1
    transition_order: int = 1
    emission_order: int = 1
    observation_dependency: int = 0

    latent_cls: type[VectorState] = VectorState
    observation_cls: type[VectorObservation] = VectorObservation
    parameter_cls: type[LGSSMParameters] = LGSSMParameters
    condition_cls: type[NoCondition] = NoCondition

    latent_context: typing.Callable[..., LatentContext[VectorState]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[VectorObservation]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[NoCondition]] = condition_context

    @staticmethod
    def prior_sample(
        key: PRNGKeyArray,
        conditions: ConditionContext[NoCondition],
        parameters: LGSSMParameters,
    ) -> LatentContext[VectorState]:
        """Sample the initial latent state.

        Default choice:
            x0 ~ N(0, Q)
        where Q = transition_noise_cholesky @ transition_noise_cholesky.T
        """
        _ = conditions
        dim = parameters.transition_matrix.shape[0]
        mean = jnp.zeros((dim,), dtype=parameters.transition_matrix.dtype)
        x0 = _mvn_sample(
            key,
            mean=mean,
            chol=parameters.transition_noise_cholesky,
        )
        return latent_context((VectorState(x=x0),))

    @staticmethod
    def prior_log_prob(
        latent: LatentContext[VectorState],
        conditions: ConditionContext[NoCondition],
        parameters: LGSSMParameters,
    ) -> Scalar:
        """Evaluate the initial latent prior log-density."""
        _ = conditions
        dim = parameters.transition_matrix.shape[0]
        mean = jnp.zeros((dim,), dtype=parameters.transition_matrix.dtype)
        return _mvn_log_prob(
            latent[0].x,
            mean=mean,
            chol=parameters.transition_noise_cholesky,
        )

    @staticmethod
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
        x = _mvn_sample(
            key,
            mean=mean,
            chol=parameters.transition_noise_cholesky,
        )
        return VectorState(x=x)

    @staticmethod
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
        return _mvn_log_prob(
            latent.x,
            mean=mean,
            chol=parameters.transition_noise_cholesky,
        )

    @staticmethod
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
        y = _mvn_sample(
            key,
            mean=mean,
            chol=parameters.emission_noise_cholesky,
        )
        return VectorObservation(y=y)

    @staticmethod
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
        return _mvn_log_prob(
            observation.y,
            mean=mean,
            chol=parameters.emission_noise_cholesky,
        )


lgssm_model = validate_sequential_model(LGSSMModel())
