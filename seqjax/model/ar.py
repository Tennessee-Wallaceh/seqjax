"""AR(1) example model implementations."""

from dataclasses import field
from typing import ClassVar
from functools import partial
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    Emission,
    ParameterPrior,
    Prior,
    SequentialModel,
    BayesianSequentialModel,
    Transition,
)
from seqjax.model.typing import (
    Condition,
    HyperParameters,
    Observation,
    Parameters,
    Particle,
)


class LatentValue(Particle):
    """Latent AR state."""

    x: Scalar

    _shape_template: ClassVar = OrderedDict(
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


class AROnlyParameters(Parameters):
    """Just the AR parameter of the AR(1) model."""

    ar: Scalar = field(default_factory=lambda: jnp.array(0.5))

    _shape_template: ClassVar = OrderedDict(
        ar=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class NoisyEmission(Observation):
    """Observation wrapping a scalar value."""

    y: Scalar

    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class HalfCauchyStds(ParameterPrior[ARParameters, HyperParameters]):
    """Half-Cauchy priors for the model standard deviations."""

    @staticmethod
    def sample(key: PRNGKeyArray, _hyperparameters: HyperParameters) -> ARParameters:
        ar_key, o_std_key, t_std_key = jrandom.split(key, 3)
        return ARParameters(
            ar=jrandom.uniform(ar_key, minval=-1, maxval=1),
            observation_std=jnp.abs(jrandom.cauchy(o_std_key)),
            transition_std=jnp.abs(jrandom.cauchy(t_std_key)),
        )

    @staticmethod
    def log_prob(
        parameteters: ARParameters, _hyperparameters: HyperParameters
    ) -> Scalar:
        """Evaluate the log-density of ``parameteters`` under the prior."""
        log_p_theta = jstats.uniform.logpdf(parameteters.ar, loc=-1.0, scale=2.0)
        log_2 = jnp.log(jnp.array(2.0))
        log_p_theta += jstats.cauchy.logpdf(parameteters.observation_std) + log_2
        log_p_theta += jstats.cauchy.logpdf(parameteters.transition_std) + log_2
        return log_p_theta


class AROnlyPrior(ParameterPrior[AROnlyParameters, HyperParameters]):
    @staticmethod
    def sample(
        key: PRNGKeyArray, _hyperparameters: HyperParameters
    ) -> AROnlyParameters:
        return AROnlyParameters(
            ar=jrandom.uniform(key, minval=-1, maxval=1),
        )

    @staticmethod
    def log_prob(
        parameteters: AROnlyParameters, _hyperparameters: HyperParameters
    ) -> Scalar:
        """Evaluate the log-density of ``parameteters`` under the prior."""
        log_p_theta = jstats.uniform.logpdf(parameteters.ar, loc=-1.0, scale=2.0)
        return log_p_theta


class InitialValue(Prior[tuple[LatentValue], None, ARParameters]):
    """Gaussian prior over the initial latent state."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: None,
        parameters: ARParameters,
    ) -> tuple[LatentValue]:
        """Sample the initial latent value."""
        stationary_scale = jnp.sqrt(
            jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
        )
        x0 = stationary_scale * jrandom.normal(
            key,
        )
        return (LatentValue(x=x0),)

    @staticmethod
    def log_prob(
        particle: tuple[LatentValue],
        conditions: None,
        parameters: ARParameters,
    ) -> Scalar:
        """Evaluate the prior log-density."""
        stationary_scale = jnp.sqrt(
            jnp.square(parameters.transition_std) / (1 - jnp.square(parameters.ar))
        )
        return jstats.norm.logpdf(particle[0].x, scale=stationary_scale)


class ARRandomWalk(Transition[LatentValue, tuple[LatentValue], None, ARParameters]):
    """Gaussian AR(1) transition."""

    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[LatentValue],
        condition: None,
        parameters: ARParameters,
    ) -> LatentValue:
        """Sample the next latent state."""
        (last_particle,) = particle_history
        next_x = (
            parameters.ar * last_particle.x
            + jrandom.normal(key) * parameters.transition_std
        )
        return LatentValue(x=next_x)

    @staticmethod
    def log_prob(
        particle_history: tuple[LatentValue],
        particle: LatentValue,
        condition: None,
        parameters: ARParameters,
    ) -> Scalar:
        """Return the transition log-density."""
        (last_particle,) = particle_history
        return jstats.norm.logpdf(
            particle.x,
            loc=parameters.ar * last_particle.x,
            scale=parameters.transition_std,
        )


class AREmission(
    Emission[tuple[LatentValue], NoisyEmission, tuple[()], None, ARParameters]
):
    """Normal emission from the latent state."""

    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(  # type: ignore[override]
        key: PRNGKeyArray,
        particle: tuple[LatentValue],
        observation_history: tuple[()],
        condition: Condition,
        parameters: ARParameters,
    ) -> NoisyEmission:
        """Sample an observation."""
        (current_particle,) = particle
        y = current_particle.x + jrandom.normal(key) * parameters.observation_std
        return NoisyEmission(y=y)

    @staticmethod
    def log_prob(  # type: ignore[override]
        particle: tuple[LatentValue],
        observation_history: tuple[()],
        observation: NoisyEmission,
        condition: Condition,
        parameters: ARParameters,
    ) -> Scalar:
        """Return the emission log-density."""
        (current_particle,) = particle
        return jstats.norm.logpdf(
            observation.y,
            loc=current_particle.x,
            scale=parameters.observation_std,
        )


class AR1Target(
    SequentialModel[
        LatentValue,
        tuple[LatentValue],
        NoisyEmission,
        tuple[()],
        None,
        None,
        ARParameters,
    ]
):
    particle_cls = LatentValue
    observation_cls = NoisyEmission
    parameter_cls = ARParameters
    prior = InitialValue()
    transition = ARRandomWalk()
    emission = AREmission()


def fill_parameter(ar_only: AROnlyParameters, ref_params: ARParameters) -> ARParameters:
    return ARParameters(
        ar_only.ar,
        observation_std=jnp.ones_like(ar_only.ar) * ref_params.observation_std,
        transition_std=jnp.ones_like(ar_only.ar) * ref_params.transition_std,
    )


class AR1Bayesian(
    BayesianSequentialModel[
        LatentValue,
        tuple[LatentValue],
        NoisyEmission,
        tuple[()],
        None,
        None,
        ARParameters,
        AROnlyParameters,
        HyperParameters,
    ]
):
    def __init__(self, ref_params: ARParameters):
        self.target_parameter = staticmethod(
            partial(fill_parameter, ref_params=ref_params)
        )

    inference_parameter_cls = AROnlyParameters
    target = AR1Target()  # defind for ARParameters
    parameter_prior = AROnlyPrior()  # defined for the partial parameters
