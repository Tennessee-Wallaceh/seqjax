"""Dimension-generic linear Gaussian state-space model on the protocol interface."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache, partial
import types
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
from jaxtyping import Array, PRNGKeyArray, Scalar

from seqjax.model.interface import (
    ConditionContext,
    LatentContext,
    ObservationContext,
    SequentialModelProtocol,
    validate_sequential_model,
)
from seqjax.model.typing import Latent, NoCondition, Observation, Parameters

DEFAULT_DIM = 5


def _validate_dim(dim: int) -> int:
    if not isinstance(dim, int):
        raise TypeError(f"dim must be an int, got {type(dim).__name__}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    return dim


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


class _VectorStateBase(Latent, abstract=True):
    dim: typing.ClassVar[int]
    x: Array

    @classmethod
    def _shape_template_from_class_args(
        cls,
        **class_kwargs: typing.Any,
    ) -> OrderedDict[str, jax.ShapeDtypeStruct]:
        dim = _validate_dim(class_kwargs["dim"])
        return OrderedDict(
            x=jax.ShapeDtypeStruct(shape=(dim,), dtype=jnp.float32),
        )


class _VectorObservationBase(Observation, abstract=True):
    dim: typing.ClassVar[int]
    y: Array

    @classmethod
    def _shape_template_from_class_args(
        cls,
        **class_kwargs: typing.Any,
    ) -> OrderedDict[str, jax.ShapeDtypeStruct]:
        dim = _validate_dim(class_kwargs["dim"])
        return OrderedDict(
            y=jax.ShapeDtypeStruct(shape=(dim,), dtype=jnp.float32),
        )


class _LGSSMParametersBase(Parameters, abstract=True):
    dim: typing.ClassVar[int]
    transition_matrix: Array
    transition_noise_cholesky: Array
    emission_matrix: Array
    emission_noise_cholesky: Array

    @classmethod
    def _shape_template_from_class_args(
        cls,
        **class_kwargs: typing.Any,
    ) -> OrderedDict[str, jax.ShapeDtypeStruct]:
        dim = _validate_dim(class_kwargs["dim"])
        return OrderedDict(
            transition_matrix=jax.ShapeDtypeStruct(shape=(dim, dim), dtype=jnp.float32),
            transition_noise_cholesky=jax.ShapeDtypeStruct(shape=(dim, dim), dtype=jnp.float32),
            emission_matrix=jax.ShapeDtypeStruct(shape=(dim, dim), dtype=jnp.float32),
            emission_noise_cholesky=jax.ShapeDtypeStruct(shape=(dim, dim), dtype=jnp.float32),
        )

    @property
    def transition_noise_covariance(self) -> Array:
        L = self.transition_noise_cholesky
        return L @ jnp.swapaxes(L, -1, -2)

    @property
    def emission_noise_covariance(self) -> Array:
        L = self.emission_noise_cholesky
        return L @ jnp.swapaxes(L, -1, -2)


@lru_cache(maxsize=None)
def make_vector_state_cls(dim: int) -> type[_VectorStateBase]:
    dim = _validate_dim(dim)

    def exec_body(ns: dict[str, typing.Any]) -> None:
        ns["__module__"] = __name__

    return typing.cast(
        type[_VectorStateBase],
        types.new_class(
            f"VectorState{dim}D",
            (_VectorStateBase,),
            kwds={"dim": dim},
            exec_body=exec_body,
        ),
    )


@lru_cache(maxsize=None)
def make_vector_observation_cls(dim: int) -> type[_VectorObservationBase]:
    dim = _validate_dim(dim)

    def exec_body(ns: dict[str, typing.Any]) -> None:
        ns["__module__"] = __name__

    return typing.cast(
        type[_VectorObservationBase],
        types.new_class(
            f"VectorObservation{dim}D",
            (_VectorObservationBase,),
            kwds={"dim": dim},
            exec_body=exec_body,
        ),
    )


@lru_cache(maxsize=None)
def make_lgssm_parameters_cls(dim: int) -> type[_LGSSMParametersBase]:
    dim = _validate_dim(dim)

    def exec_body(ns: dict[str, typing.Any]) -> None:
        ns["__module__"] = __name__
        ns["__annotations__"] = {
            "transition_matrix": Array,
            "transition_noise_cholesky": Array,
            "emission_matrix": Array,
            "emission_noise_cholesky": Array,
        }
        ns["transition_matrix"] = field(
            default_factory=lambda dim=dim: 0.7 * jnp.eye(dim)
        )
        ns["transition_noise_cholesky"] = field(
            default_factory=lambda dim=dim: jnp.eye(dim)
        )
        ns["emission_matrix"] = field(
            default_factory=lambda dim=dim: jnp.eye(dim)
        )
        ns["emission_noise_cholesky"] = field(
            default_factory=lambda dim=dim: jnp.eye(dim)
        )

    return typing.cast(
        type[_LGSSMParametersBase],
        types.new_class(
            f"LGSSMParameters{dim}D",
            (_LGSSMParametersBase,),
            kwds={"dim": dim},
            exec_body=exec_body,
        ),
    )


latent_context: typing.Callable[[tuple[_VectorStateBase]], LatentContext[_VectorStateBase]]
latent_context = partial(LatentContext, length=1)
observation_context: typing.Callable[[tuple], ObservationContext[_VectorObservationBase]]
observation_context = partial(ObservationContext, length=0)
condition_context: typing.Callable[[tuple], ConditionContext[NoCondition]]
condition_context = partial(ConditionContext, length=0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LGSSMModel(
    SequentialModelProtocol[
        _VectorStateBase,
        _VectorObservationBase,
        NoCondition,
        _LGSSMParametersBase,
    ]
):
    latent_cls: type[_VectorStateBase]
    observation_cls: type[_VectorObservationBase]
    parameter_cls: type[_LGSSMParametersBase]
    condition_cls: type[NoCondition] = NoCondition

    prior_order: int = 1
    transition_order: int = 1
    emission_order: int = 1
    observation_dependency: int = 0

    latent_context: typing.Callable[..., LatentContext[_VectorStateBase]] = latent_context
    observation_context: typing.Callable[..., ObservationContext[_VectorObservationBase]] = observation_context
    condition_context: typing.Callable[..., ConditionContext[NoCondition]] = condition_context

    @property
    def dim(self) -> int:
        return self.parameter_cls.dim

    @staticmethod
    def _validate_parameter_shapes(parameters: _LGSSMParametersBase) -> int:
        dim = parameters.transition_matrix.shape[0]
        expected = (dim, dim)
        field_shapes = {
            "transition_matrix": parameters.transition_matrix.shape,
            "transition_noise_cholesky": parameters.transition_noise_cholesky.shape,
            "emission_matrix": parameters.emission_matrix.shape,
            "emission_noise_cholesky": parameters.emission_noise_cholesky.shape,
        }
        for field_name, shape in field_shapes.items():
            if shape != expected:
                raise ValueError(
                    f"{field_name} must have shape {expected}, got {shape}"
                )
        return dim

    @staticmethod
    def prior_sample(
        key: PRNGKeyArray,
        conditions: ConditionContext[NoCondition],
        parameters: _LGSSMParametersBase,
    ) -> LatentContext[_VectorStateBase]:
        _ = conditions
        dim = LGSSMModel._validate_parameter_shapes(parameters)
        mean = jnp.zeros((dim,), dtype=parameters.transition_matrix.dtype)
        x0 = _mvn_sample(
            key,
            mean=mean,
            chol=parameters.transition_noise_cholesky,
        )
        latent_cls = make_vector_state_cls(dim)
        return latent_context((latent_cls(x=x0),))

    @staticmethod
    def prior_log_prob(
        latent: LatentContext[_VectorStateBase],
        conditions: ConditionContext[NoCondition],
        parameters: _LGSSMParametersBase,
    ) -> Scalar:
        _ = conditions
        dim = LGSSMModel._validate_parameter_shapes(parameters)
        mean = jnp.zeros((dim,), dtype=parameters.transition_matrix.dtype)
        return _mvn_log_prob(
            latent[0].x,
            mean=mean,
            chol=parameters.transition_noise_cholesky,
        )

    @staticmethod
    def transition_sample(
        key: PRNGKeyArray,
        latent_history: LatentContext[_VectorStateBase],
        condition: NoCondition,
        parameters: _LGSSMParametersBase,
    ) -> _VectorStateBase:
        _ = condition
        LGSSMModel._validate_parameter_shapes(parameters)
        last_state = latent_history[0]
        mean = parameters.transition_matrix @ last_state.x
        x = _mvn_sample(
            key,
            mean=mean,
            chol=parameters.transition_noise_cholesky,
        )
        latent_cls = make_vector_state_cls(parameters.transition_matrix.shape[0])
        return latent_cls(x=x)

    @staticmethod
    def transition_log_prob(
        latent_history: LatentContext[_VectorStateBase],
        latent: _VectorStateBase,
        condition: NoCondition,
        parameters: _LGSSMParametersBase,
    ) -> Scalar:
        _ = condition
        LGSSMModel._validate_parameter_shapes(parameters)
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
        latent_history: LatentContext[_VectorStateBase],
        observation_history: ObservationContext[_VectorObservationBase],
        condition: NoCondition,
        parameters: _LGSSMParametersBase,
    ) -> _VectorObservationBase:
        _ = (observation_history, condition)
        LGSSMModel._validate_parameter_shapes(parameters)
        state = latent_history[0]
        mean = parameters.emission_matrix @ state.x
        y = _mvn_sample(
            key,
            mean=mean,
            chol=parameters.emission_noise_cholesky,
        )
        observation_cls = make_vector_observation_cls(parameters.emission_matrix.shape[0])
        return observation_cls(y=y)

    @staticmethod
    def emission_log_prob(
        latent_history: LatentContext[_VectorStateBase],
        observation: _VectorObservationBase,
        observation_history: ObservationContext[_VectorObservationBase],
        condition: NoCondition,
        parameters: _LGSSMParametersBase,
    ) -> Scalar:
        _ = (observation_history, condition)
        LGSSMModel._validate_parameter_shapes(parameters)
        state = latent_history[0]
        mean = parameters.emission_matrix @ state.x
        return _mvn_log_prob(
            observation.y,
            mean=mean,
            chol=parameters.emission_noise_cholesky,
        )


@lru_cache(maxsize=None)
def lgssm(dim: int = DEFAULT_DIM) -> LGSSMModel:
    dim = _validate_dim(dim)
    return validate_sequential_model(
        LGSSMModel(
            latent_cls=make_vector_state_cls(dim),
            observation_cls=make_vector_observation_cls(dim),
            parameter_cls=make_lgssm_parameters_cls(dim),
        )
    )


VectorState5D = make_vector_state_cls(DEFAULT_DIM)
VectorObservation5D = make_vector_observation_cls(DEFAULT_DIM)
LGSSMParameters5D = make_lgssm_parameters_cls(DEFAULT_DIM)

VectorState = VectorState5D
VectorObservation = VectorObservation5D
LGSSMParameters = LGSSMParameters5D
lgssm_model = lgssm(DEFAULT_DIM)
