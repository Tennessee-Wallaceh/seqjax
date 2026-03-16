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
    validate_sequential_model,
    ConditionContext,
    LatentContext,
    ObservationContext,
    ParameterizationProtocol,
    SequentialModelProtocol,
)
from seqjax.model.typing import HyperParameters, Parameters, Latent, Observation, NoCondition, NoHyper


prior_order = 1
transition_order = 1
emission_order = 1
observation_dependency = 0


@dataclass
class RoughLatentVar(Latent):
    """Latent multifactor rough-style log-variance state."""
    z: jax.Array
    _shape_template: typing.ClassVar = OrderedDict(
        z=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.float32),
    )


@dataclass
class LogReturnObs(Observation):
    """Observed log return."""
    log_return: Scalar
    _shape_template: typing.ClassVar = OrderedDict(
        log_return=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


@dataclass
class RoughLogVarParams(Parameters):
    """
    Model parameters.

    long_term_log_var:
        Baseline log variance level.

    roughness:
        Hurst-like roughness parameter H in (0, 0.5).

    shared_scale:
        Global scale for the shared latent innovation.
    """
    long_term_log_var: Scalar
    roughness: Scalar
    shared_scale: Scalar
    _shape_template: typing.ClassVar = OrderedDict(
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        roughness=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        shared_scale=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


@dataclass
class RoughVolHyper(HyperParameters):
    """
    Fixed hyperparameters defining the factor approximation.

    decay_scales:
        Positive decay rates lambda_i.

    idio_scale:
        Small idiosyncratic innovation std for each factor.
        This keeps the transition full-rank.

    dt:
        Discrete timestep.

    rough_weight_power:
        Controls how strongly H tilts weight mass toward fast vs slow factors.
        This is a pragmatic rough-style approximation, not a literal quadrature rule.
    """
    decay_scales: jax.Array
    idio_scale: jax.Array
    dt: Scalar = field(default_factory=lambda: jnp.array(1.0))
    rough_weight_power: Scalar = field(default_factory=lambda: jnp.array(1.0))

    _shape_template: typing.ClassVar = OrderedDict(
        decay_scales=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.float32),
        idio_scale=jax.ShapeDtypeStruct(shape=(8,), dtype=jnp.float32),
        dt=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        rough_weight_power=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


latent_cls = RoughLatentVar
observation_cls = LogReturnObs
parameter_cls = RoughLogVarParams
condition_cls = NoCondition

latent_context = partial(LatentContext, length=transition_order)
observation_context = partial(ObservationContext, length=observation_dependency)
condition_context = partial(ConditionContext, length=0)


def _n_factors(hyperparameters: RoughVolHyper) -> int:
    return hyperparameters.decay_scales.shape[0]


def _factor_ar_coefficients(
    hyperparameters: RoughVolHyper,
) -> jax.Array:
    """phi_i = exp(-lambda_i * dt)"""
    return jnp.exp(-hyperparameters.decay_scales * hyperparameters.dt)


def _normalized_factor_weights(
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> jax.Array:
    """
    Pragmatic rough-style weighting profile across timescales.

    Smaller H -> relatively more mass on fast factors.
    Larger H -> flatter spectrum.

    We normalize to unit l2 norm so shared_scale controls overall amplitude.
    """
    decay_scales = hyperparameters.decay_scales

    # Map H in (0, 0.5) to a positive tilt exponent.
    # This is intentionally simple and interpretable rather than a literal kernel quadrature.
    tilt = hyperparameters.rough_weight_power * (0.5 - parameters.roughness)

    raw_weights = jnp.power(decay_scales, tilt)
    norm = jnp.linalg.norm(raw_weights)
    return raw_weights / norm


def _shared_innovation_loadings(
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> jax.Array:
    return parameters.shared_scale * _normalized_factor_weights(parameters, hyperparameters)


def _log_var_from_latent(
    latent: RoughLatentVar,
    parameters: RoughLogVarParams,
) -> Scalar:
    return parameters.long_term_log_var + jnp.sum(latent.z)


def _stationary_factor_std_approx(
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> jax.Array:
    """
    Approximate per-factor stationary std ignoring cross-factor covariance from shared shocks.
    Fine as a simple prior default.
    """
    phi = _factor_ar_coefficients(hyperparameters)
    shared_loadings = _shared_innovation_loadings(parameters, hyperparameters)
    total_factor_var = jnp.square(shared_loadings) + jnp.square(hyperparameters.idio_scale)
    return jnp.sqrt(total_factor_var / (1.0 - jnp.square(phi)))


def prior_sample(
    key: PRNGKeyArray,
    conditions: ConditionContext[NoCondition],
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> LatentContext[RoughLatentVar]:
    _ = conditions
    sigma = _stationary_factor_std_approx(parameters, hyperparameters)
    z0 = sigma * jrandom.normal(key, shape=sigma.shape)
    start_latent = RoughLatentVar(z=z0)
    return latent_context((start_latent,))


def prior_log_prob(
    latent: LatentContext[RoughLatentVar],
    conditions: ConditionContext[NoCondition],
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> Scalar:
    _ = conditions
    sigma = _stationary_factor_std_approx(parameters, hyperparameters)
    return jnp.sum(jstats.norm.logpdf(latent[0].z, loc=0.0, scale=sigma))


def transition_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[RoughLatentVar],
    condition: NoCondition,
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> RoughLatentVar:
    _ = condition
    last_latent = latent_history[0]

    key_shared, key_idio = jrandom.split(key)
    eps_shared = jrandom.normal(key_shared)
    eps_idio = jrandom.normal(key_idio, shape=last_latent.z.shape)

    phi = _factor_ar_coefficients(hyperparameters)
    shared_loadings = _shared_innovation_loadings(parameters, hyperparameters)

    next_z = (
        phi * last_latent.z
        + shared_loadings * eps_shared
        + hyperparameters.idio_scale * eps_idio
    )
    return RoughLatentVar(z=next_z)


def transition_log_prob(
    latent_history: LatentContext[RoughLatentVar],
    latent: RoughLatentVar,
    condition: NoCondition,
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> Scalar:
    _ = condition
    last_latent = latent_history[0]

    phi = _factor_ar_coefficients(hyperparameters)
    shared_loadings = _shared_innovation_loadings(parameters, hyperparameters)
    idio_var = jnp.square(hyperparameters.idio_scale)

    loc = phi * last_latent.z
    cov = (
        jnp.outer(shared_loadings, shared_loadings)
        + jnp.diag(idio_var)
    )

    diff = latent.z - loc
    chol = jnp.linalg.cholesky(cov)
    solve = jax.scipy.linalg.solve_triangular(chol, diff, lower=True)
    quad = jnp.sum(jnp.square(solve))
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = diff.shape[0]

    return -0.5 * (
        dim * jnp.log(2.0 * jnp.pi)
        + logdet
        + quad
    )


def emission_sample(
    key: PRNGKeyArray,
    latent_history: LatentContext[RoughLatentVar],
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> LogReturnObs:
    _ = observation_history
    _ = condition
    _ = hyperparameters

    current_latent = latent_history[0]
    current_log_var = _log_var_from_latent(current_latent, parameters)
    return_scale = jnp.exp(0.5 * current_log_var)

    return LogReturnObs(
        log_return=jrandom.normal(key) * return_scale
    )


def emission_log_prob(
    latent_history: LatentContext[RoughLatentVar],
    observation: LogReturnObs,
    observation_history: ObservationContext[LogReturnObs],
    condition: NoCondition,
    parameters: RoughLogVarParams,
    hyperparameters: RoughVolHyper,
) -> Scalar:
    _ = observation_history
    _ = condition
    _ = hyperparameters

    current_latent = latent_history[0]
    current_log_var = _log_var_from_latent(current_latent, parameters)
    return_scale = jnp.exp(0.5 * current_log_var)

    return jstats.norm.logpdf(
        observation.log_return,
        loc=0.0,
        scale=return_scale,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RoughStochasticVar(
    SequentialModelProtocol[
        RoughLatentVar,
        LogReturnObs,
        NoCondition,
        RoughLogVarParams,
    ]
):
    prior_order: int = prior_order
    transition_order: int = transition_order
    emission_order: int = emission_order
    observation_dependency: int = observation_dependency

    latent_cls: type[RoughLatentVar] = latent_cls
    observation_cls: type[LogReturnObs] = observation_cls
    parameter_cls: type[RoughLogVarParams] = parameter_cls
    condition_cls: type[NoCondition] = condition_cls

    latent_context: typing.Callable[..., LatentContext[RoughLatentVar]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[LogReturnObs]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[NoCondition]] = condition_context

    hyperparameters: RoughVolHyper = field(
        default_factory=lambda: RoughVolHyper(
            decay_scales=jnp.exp(jnp.linspace(jnp.log(1e-3), jnp.log(1e-1), 8)),
            idio_scale=0.01 * jnp.ones((8,)),
        )
    )

    def prior_sample(
        self,
        key: PRNGKeyArray,
        conditions: ConditionContext[NoCondition],
        parameters: RoughLogVarParams,
    ) -> LatentContext[RoughLatentVar]:
        return prior_sample(key, conditions, parameters, self.hyperparameters)

    def prior_log_prob(
        self,
        latent: LatentContext[RoughLatentVar],
        conditions: ConditionContext[NoCondition],
        parameters: RoughLogVarParams,
    ) -> Scalar:
        return prior_log_prob(latent, conditions, parameters, self.hyperparameters)

    def transition_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[RoughLatentVar],
        condition: NoCondition,
        parameters: RoughLogVarParams,
    ) -> RoughLatentVar:
        return transition_sample(
            key,
            latent_history,
            condition,
            parameters,
            self.hyperparameters,
        )

    def transition_log_prob(
        self,
        latent_history: LatentContext[RoughLatentVar],
        latent: RoughLatentVar,
        condition: NoCondition,
        parameters: RoughLogVarParams,
    ) -> Scalar:
        return transition_log_prob(
            latent_history,
            latent,
            condition,
            parameters,
            self.hyperparameters,
        )

    def emission_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[RoughLatentVar],
        observation_history: ObservationContext[LogReturnObs],
        condition: NoCondition,
        parameters: RoughLogVarParams,
    ) -> LogReturnObs:
        return emission_sample(
            key,
            latent_history,
            observation_history,
            condition,
            parameters,
            self.hyperparameters,
        )

    def emission_log_prob(
        self,
        latent_history: LatentContext[RoughLatentVar],
        observation: LogReturnObs,
        observation_history: ObservationContext[LogReturnObs],
        condition: NoCondition,
        parameters: RoughLogVarParams,
    ) -> Scalar:
        return emission_log_prob(
            latent_history,
            observation,
            observation_history,
            condition,
            parameters,
            self.hyperparameters,
        )


@dataclass
class UncRoughLogVarParams(Parameters):
    sft_inv_shared_scale: Scalar
    roughness_raw: Scalar
    long_term_log_var: Scalar
    _shape_template: typing.ClassVar = OrderedDict(
        sft_inv_shared_scale=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        roughness_raw=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        long_term_log_var=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


@jax.tree_util.register_dataclass
@dataclass
class RoughVarParameterization(
    ParameterizationProtocol[
        RoughLogVarParams,
        UncRoughLogVarParams,
        RoughVolHyper,
    ]
):
    hyperparameters: RoughVolHyper 
    inference_parameter_cls: type[UncRoughLogVarParams] = UncRoughLogVarParams

    def to_model_parameters(
        self,
        inference_parameters: UncRoughLogVarParams,
    ) -> RoughLogVarParams:
        roughness = 0.5 * jax.nn.sigmoid(inference_parameters.roughness_raw)
        return RoughLogVarParams(
            shared_scale=jax.nn.softplus(inference_parameters.sft_inv_shared_scale),
            roughness=roughness,
            long_term_log_var=inference_parameters.long_term_log_var,
        )

    def from_model_parameters(
        self,
        model_parameters: RoughLogVarParams,
    ) -> UncRoughLogVarParams:
        roughness_ratio = 2.0 * model_parameters.roughness
        return UncRoughLogVarParams(
            sft_inv_shared_scale=jnp.log(jnp.expm1(model_parameters.shared_scale)),
            roughness_raw=jnp.log(roughness_ratio) - jnp.log1p(-roughness_ratio),
            long_term_log_var=model_parameters.long_term_log_var,
        )

    def sample(
        self,
        key: PRNGKeyArray,
    ) -> UncRoughLogVarParams:
        long_term_log_var_mean = 2 * jnp.log(jnp.array(0.16))
        k1, k2, k3 = jrandom.split(key, 3)

        model_parameters = RoughLogVarParams(
            shared_scale=0.05 + 0.10 * jrandom.uniform(k1),
            roughness=0.02 + 0.46 * jrandom.uniform(k2),
            long_term_log_var=long_term_log_var_mean + jrandom.normal(k3),
        )
        return self.from_model_parameters(model_parameters)

    def log_prob(
        self,
        inference_parameters: UncRoughLogVarParams,
    ) -> Scalar:
        model_params = self.to_model_parameters(inference_parameters)
        long_term_log_var_mean = 2 * jnp.log(jnp.array(0.16))

        lad_shared_scale = jax.nn.log_sigmoid(inference_parameters.sft_inv_shared_scale)
        roughness_ratio = 2.0 * model_params.roughness
        lad_roughness = jnp.log(roughness_ratio) + jnp.log1p(-roughness_ratio)

        return (
            jstats.gamma.logpdf(1.0 / model_params.shared_scale, 10.0, scale=1.0 / 10.0)
            + lad_shared_scale
            + jstats.uniform.logpdf(model_params.roughness, loc=0.0, scale=0.5)
            + lad_roughness
            + jstats.norm.logpdf(
                model_params.long_term_log_var,
                loc=long_term_log_var_mean,
                scale=1.0,
            )
        )


@jax.tree_util.register_dataclass
@dataclass
class RoughStochasticVarBayesian:
    target: RoughStochasticVar
    parameterization: RoughVarParameterization


def rough_stochastic_var(
    n_factors: int = 8,
    dt: float = 1.0,
    min_decay: float = 1e-3,
    max_decay: float = 1e-1,
    idio_scale: float = 0.01,
    rough_weight_power: float = 1.0,
) -> RoughStochasticVarBayesian:
    decay_scales = jnp.exp(
        jnp.linspace(jnp.log(min_decay), jnp.log(max_decay), n_factors)
    )
    hyperparameters = RoughVolHyper(
        decay_scales=decay_scales,
        idio_scale=idio_scale * jnp.ones((n_factors,)),
        dt=jnp.array(dt),
        rough_weight_power=jnp.array(rough_weight_power),
    )
    target = validate_sequential_model(
        RoughStochasticVar(hyperparameters=hyperparameters)
    )
    return RoughStochasticVarBayesian(
        target=target,
        parameterization=RoughVarParameterization(
            hyperparameters=hyperparameters
        ),
    )