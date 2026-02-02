"""Multidimensional linear Gaussian state space model."""

from collections import OrderedDict
from dataclasses import field
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, PRNGKeyArray, Scalar

from seqjax.model.base import Emission, Prior, SequentialModel, Transition
from seqjax.model.typing import Observation, Parameters, Latent, NoCondition


class VectorState50D(Latent):
    """Multivariate latent state."""

    x: Array

    _shape_template: ClassVar = OrderedDict(
        x=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
    )


class VectorObservation(Observation):
    """Vector-valued observation."""

    y: Array

    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
    )


class LGSSMParameters(Parameters):
    """Parameters of a linear Gaussian state space model."""

    transition_matrix: Array = field(default_factory=lambda: jnp.eye(2))
    transition_noise_scale: Array = field(default_factory=lambda: jnp.ones(2))
    emission_matrix: Array = field(default_factory=lambda: jnp.eye(2))
    emission_noise_scale: Array = field(default_factory=lambda: jnp.ones(2))

    _shape_template: ClassVar = OrderedDict(
        transition_matrix=jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
        transition_noise_scale=jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
        emission_matrix=jax.ShapeDtypeStruct(shape=(2, 2), dtype=jnp.float32),
        emission_noise_scale=jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32),
    )


class GaussianPrior:
    """Gaussian prior over the initial state."""

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[NoCondition],
        parameters: LGSSMParameters,
    ) -> tuple[VectorState]:
        mean = jnp.zeros_like(parameters.transition_noise_scale)
        scale = parameters.transition_noise_scale
        x0 = mean + scale * jrandom.normal(key, shape=scale.shape)
        return (VectorState(x=x0),)

    @staticmethod
    def log_prob(
        latent: tuple[VectorState],
        conditions: tuple[NoCondition],
        parameters: LGSSMParameters,
    ) -> Scalar:
        scale = parameters.transition_noise_scale
        logp = jstats.norm.logpdf(latent[0].x, loc=0.0, scale=scale)
        return logp.sum()


guassian_prior: Prior[
    tuple[VectorState],
    tuple[NoCondition],
    LGSSMParameters,
] = Prior(sample=GaussianPrior.sample, log_prob=GaussianPrior.log_prob, order=1)


class GaussianTransition:
    """Linear Gaussian state transition."""

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        latent_history: tuple[VectorState],
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> VectorState:
        (last_state,) = latent_history
        mean = parameters.transition_matrix @ last_state.x
        noise = parameters.transition_noise_scale * jrandom.normal(
            key, shape=parameters.transition_noise_scale.shape
        )
        return VectorState(x=mean + noise)

    @staticmethod
    def log_prob(
        latent_history: tuple[VectorState],
        latent: VectorState,
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> Scalar:
        (last_state,) = latent_history
        mean = parameters.transition_matrix @ last_state.x
        logp = jstats.norm.logpdf(
            latent.x, loc=mean, scale=parameters.transition_noise_scale
        )
        return logp.sum()


gaussian_transition: Transition[
    tuple[VectorState],
    VectorState,
    NoCondition,
    LGSSMParameters,
] = Transition(
    sample=GaussianTransition.sample, log_prob=GaussianTransition.log_prob, order=1
)


class GaussianEmission:
    """Gaussian emission from the latent state."""

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        latent: tuple[VectorState],
        observation_history: tuple[()],
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> VectorObservation:
        (state,) = latent
        mean = parameters.emission_matrix @ state.x
        noise = parameters.emission_noise_scale * jrandom.normal(
            key, shape=parameters.emission_noise_scale.shape
        )
        return VectorObservation(y=mean + noise)

    @staticmethod
    def log_prob(
        latent: tuple[VectorState],
        observation: VectorObservation,
        observation_history: tuple[()],
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> Scalar:
        (state,) = latent
        mean = parameters.emission_matrix @ state.x
        logp = jstats.norm.logpdf(
            observation.y, loc=mean, scale=parameters.emission_noise_scale
        )
        return logp.sum()


guassian_emission: Emission[
    tuple[VectorState],
    NoCondition,
    VectorObservation,
    LGSSMParameters,
] = Emission(
    sample=GaussianEmission.sample,
    log_prob=GaussianEmission.log_prob,
    order=1,
)


class LinearGaussianSSM(
    SequentialModel[
        VectorState,
        VectorObservation,
        NoCondition,
        LGSSMParameters,
    ]
):
    latent_cls = VectorState
    observation_cls = VectorObservation
    parameter_cls = LGSSMParameters
    prior = guassian_prior
    transition = gaussian_transition
    emission = guassian_emission
