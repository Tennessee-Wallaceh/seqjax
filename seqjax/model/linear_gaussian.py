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

from seqjax.model.base import BayesianSequentialModel, ParameterPrior, Emission, Prior, SequentialModel, Transition
from seqjax.model.typing import Observation, Parameters, Latent, NoCondition, HyperParameters
import jax.scipy.special as jspecial

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
    """Parameters of a linear Gaussian state space model (50D).

    Transition noise covariance: Q = D R D, with
      - D = diag(transition_noise_scale), transition_noise_scale > 0 elementwise
      - R = transition_noise_corr_cholesky @ transition_noise_corr_cholesky.T
        is a correlation matrix (SPD, diag == 1)

    Emission noise covariance: same structure.
    """

    transition_matrix: Array = field(default_factory=lambda: jnp.eye(50))

    # Target-space (constrained) noise parameterisation
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
        # L_Q = D L_R  => Q = L_Q L_Q^T
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
    inference_parameter_cls = LGSSMParameters
    target = LinearGaussianSSM()
    parameter_prior = LGSSMParameterPrior()
    convert_to_model_parameters = staticmethod(lambda parameters: parameters)
