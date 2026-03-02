"""
Multidimensional linear Gaussian state space model.
We use the LKJ Cholesky Cov parameterisation.
"""

from collections import OrderedDict
from dataclasses import field
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, PRNGKeyArray, Scalar

import jax.scipy.special as jspecial
from seqjax.model.base import (
    BayesianSequentialModel,
    ParameterPrior,
    SequentialModelBase,
    LatentContext,
    ObservationContext,
    ConditionContext,
)
from seqjax.model.typing import Observation, Parameters, Latent, NoCondition, HyperParameters


class VectorState50D(Latent):
    """Multivariate latent state."""

    x: Array

    _shape_template: ClassVar = OrderedDict(
        x=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
    )


class VectorObservation50D(Observation):
    """Vector-valued observation."""

    y: Array

    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
    )


def wishart_logpdf(precision_matrix, df, wishart_rate):
    L = jnp.linalg.cholesky(precision_matrix)
    dim = precision_matrix.shape[0]
    log_det_x = 2 * jnp.sum(jnp.log(L.diagonal()))
    log_det_scale = -dim * jnp.log(wishart_rate)
    trace_term = wishart_rate * jnp.trace(precision_matrix)

    return (
        0.5 * (df - dim - 1) * log_det_x
        - 0.5 * trace_term
        - 0.5 * dim * df * jnp.log(2)
        - 0.5 * df * log_det_scale
        - jspecial.multigammaln(df / 2, dim)
    )


class LGSSMParameters50D(Parameters):
    """Parameters of a linear Gaussian state space model (50D)."""

    transition_matrix: Array = field(default_factory=lambda: jnp.eye(50))
    transition_noise_scale: Array = field(default_factory=lambda: jnp.ones(50))
    transition_noise_corr_cholesky: Array = field(default_factory=lambda: jnp.eye(50))

    emission_matrix: Array = field(default_factory=lambda: jnp.eye(50))
    emission_noise_scale: Array = field(default_factory=lambda: jnp.ones(50))
    emission_noise_corr_cholesky: Array = field(default_factory=lambda: jnp.eye(50))

    _shape_template: ClassVar = OrderedDict(
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
        L_Q = self.transition_noise_cholesky
        return L_Q @ L_Q.T

    @property
    def emission_noise_cholesky(self) -> Array:
        return jnp.diag(self.emission_noise_scale) @ self.emission_noise_corr_cholesky

    @property
    def emission_noise_covariance(self) -> Array:
        L = self.emission_noise_cholesky
        return L @ L.T


LGSSMParameters = LGSSMParameters50D
VectorState = VectorState50D
VectorObservation = VectorObservation50D


class LGSSMParameterPrior(ParameterPrior[LGSSMParameters, HyperParameters]):
    """Prior over the parameters of the linear Gaussian SSM."""

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        hypers: HyperParameters,
    ) -> LGSSMParameters:
        raise NotImplementedError

    def log_prob(
        self,
        parameters: LGSSMParameters,
        hypers: HyperParameters,
    ) -> Scalar:
        raise NotImplementedError


class LinearGaussianSSM(
    SequentialModelBase[
        VectorState,
        VectorObservation,
        NoCondition,
        LGSSMParameters,
    ]
):
    latent_cls = VectorState
    observation_cls = VectorObservation
    parameter_cls = LGSSMParameters
    condition_cls = NoCondition

    prior_order = 1
    transition_order = 1
    emission_order = 1
    observation_dependency = 0

    def prior_sample(
        self,
        key: PRNGKeyArray,
        conditions: ConditionContext[NoCondition],
        parameters: LGSSMParameters,
    ) -> LatentContext[VectorState]:
        del conditions
        mean = jnp.zeros_like(parameters.transition_noise_scale)
        scale = parameters.transition_noise_scale
        x0 = mean + scale * jrandom.normal(key, shape=scale.shape)
        return self.make_latent_context(VectorState(x=x0))

    def prior_log_prob(
        self,
        latent: LatentContext[VectorState],
        conditions: ConditionContext[NoCondition],
        parameters: LGSSMParameters,
    ) -> Scalar:
        del conditions
        scale = parameters.transition_noise_scale
        logp = jstats.norm.logpdf(latent[-1].x, loc=0.0, scale=scale)
        return logp.sum()

    def transition_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[VectorState],
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> VectorState:
        del condition
        mean = parameters.transition_matrix @ latent_history[-1].x
        noise = parameters.transition_noise_scale * jrandom.normal(
            key, shape=parameters.transition_noise_scale.shape
        )
        return VectorState(x=mean + noise)

    def transition_log_prob(
        self,
        latent_history: LatentContext[VectorState],
        latent: VectorState,
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> Scalar:
        del condition
        mean = parameters.transition_matrix @ latent_history[-1].x
        logp = jstats.norm.logpdf(
            latent.x, loc=mean, scale=parameters.transition_noise_scale
        )
        return logp.sum()

    def emission_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[VectorState],
        observation_history: ObservationContext[VectorObservation],
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> VectorObservation:
        del observation_history, condition
        mean = parameters.emission_matrix @ latent_history[-1].x
        noise = parameters.emission_noise_scale * jrandom.normal(
            key, shape=parameters.emission_noise_scale.shape
        )
        return VectorObservation(y=mean + noise)

    def emission_log_prob(
        self,
        latent_history: LatentContext[VectorState],
        observation: VectorObservation,
        observation_history: ObservationContext[VectorObservation],
        condition: NoCondition,
        parameters: LGSSMParameters,
    ) -> Scalar:
        del observation_history, condition
        mean = parameters.emission_matrix @ latent_history[-1].x
        logp = jstats.norm.logpdf(
            observation.y,
            loc=mean,
            scale=parameters.emission_noise_scale,
        )
        return logp.sum()


class BayesianLinearGaussianSSM(
    BayesianSequentialModel[
        VectorState,
        VectorObservation,
        NoCondition,
        LGSSMParameters,
        LGSSMParameters,
        HyperParameters,
    ]
):
    def __init__(self, hyperparameters: HyperParameters | None = None):
        super().__init__(hyperparameters=hyperparameters)

    inference_parameter_cls = LGSSMParameters
    target = LinearGaussianSSM()
    def parameter_prior(self) -> LGSSMParameterPrior:
        return LGSSMParameterPrior()
    convert_to_model_parameters = staticmethod(lambda parameters: parameters)
