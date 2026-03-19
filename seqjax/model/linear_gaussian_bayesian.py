"""Bayesian parameterisation for an identified dimension-generic LGSSM."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
import types
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

import seqjax.model.typing as seqjtyping
from . import interface, linear_gaussian as lg_module
from .interface import ParameterizationProtocol


def _validate_dim(dim: int) -> int:
    if not isinstance(dim, int):
        raise TypeError(f"dim must be an int, got {type(dim).__name__}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    return dim


def _strictly_lower_size(dim: int) -> int:
    dim = _validate_dim(dim)
    return dim * (dim - 1) // 2


def _inverse_softplus(x: jax.Array) -> jax.Array:
    if jnp.any(x <= 0):
        raise ValueError("All scale parameters must be strictly positive")
    return jnp.log(jnp.expm1(x))


def _diag_cholesky(scale: jax.Array) -> jax.Array:
    return jnp.diag(scale)


def _fill_strictly_lower_triangular(vec: jax.Array, dim: int) -> jax.Array:
    """Fill the strictly lower-triangular part of a matrix from a vector."""
    expected_size = _strictly_lower_size(dim)
    if vec.shape != (expected_size,):
        raise ValueError(
            "strictly lower-triangular vector must have shape "
            f"({expected_size},), got {vec.shape}"
        )
    mat = jnp.zeros((dim, dim), dtype=vec.dtype)
    row_ix, col_ix = jnp.tril_indices(dim, k=-1)
    return mat.at[row_ix, col_ix].set(vec)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LGSSMHyperParameters:
    dim: int = field(default=lg_module.DEFAULT_DIM, metadata=dict(static=True))

    def __post_init__(self) -> None:
        _validate_dim(self.dim)


class _UncLGSSMParametersBase(seqjtyping.Parameters, abstract=True):
    dim: typing.ClassVar[int]
    unc_transition_diag: jax.Array
    emission_strictly_lower: jax.Array
    emission_log_diag: jax.Array
    sft_inv_emission_noise_scale: jax.Array

    @classmethod
    def _shape_template_from_class_args(
        cls,
        **class_kwargs: typing.Any,
    ) -> OrderedDict[str, jax.ShapeDtypeStruct]:
        dim = _validate_dim(class_kwargs["dim"])
        lower_size = _strictly_lower_size(dim)
        return OrderedDict(
            unc_transition_diag=jax.ShapeDtypeStruct(shape=(dim,), dtype=jnp.float32),
            emission_strictly_lower=jax.ShapeDtypeStruct(shape=(lower_size,), dtype=jnp.float32),
            emission_log_diag=jax.ShapeDtypeStruct(shape=(dim,), dtype=jnp.float32),
            sft_inv_emission_noise_scale=jax.ShapeDtypeStruct(shape=(dim,), dtype=jnp.float32),
        )


@lru_cache(maxsize=None)
def make_unc_lgssm_parameters_cls(dim: int) -> type[_UncLGSSMParametersBase]:
    dim = _validate_dim(dim)
    lower_size = _strictly_lower_size(dim)

    def exec_body(ns: dict[str, typing.Any]) -> None:
        ns["__module__"] = __name__
        ns["__annotations__"] = {
            "unc_transition_diag": jax.Array,
            "emission_strictly_lower": jax.Array,
            "emission_log_diag": jax.Array,
            "sft_inv_emission_noise_scale": jax.Array,
        }
        ns["unc_transition_diag"] = field(
            default_factory=lambda dim=dim: jnp.zeros(dim)
        )
        ns["emission_strictly_lower"] = field(
            default_factory=lambda lower_size=lower_size: jnp.zeros(lower_size)
        )
        ns["emission_log_diag"] = field(
            default_factory=lambda dim=dim: jnp.zeros(dim)
        )
        ns["sft_inv_emission_noise_scale"] = field(
            default_factory=lambda dim=dim: jnp.zeros(dim)
        )

    return typing.cast(
        type[_UncLGSSMParametersBase],
        types.new_class(
            f"UncLGSSMParameters{dim}D",
            (_UncLGSSMParametersBase,),
            kwds={"dim": dim},
            exec_body=exec_body,
        ),
    )


@dataclass
class FullParameterization(
    ParameterizationProtocol[
        lg_module._LGSSMParametersBase,
        _UncLGSSMParametersBase,
        LGSSMHyperParameters,
    ]
):
    hyperparameters: LGSSMHyperParameters = field(default_factory=LGSSMHyperParameters)
    inference_parameter_cls: type[_UncLGSSMParametersBase] = field(init=False)

    def __post_init__(self) -> None:
        self.inference_parameter_cls = make_unc_lgssm_parameters_cls(
            self.hyperparameters.dim
        )

    @property
    def dim(self) -> int:
        return self.hyperparameters.dim

    def _validate_model_parameters(
        self,
        model_parameters: lg_module._LGSSMParametersBase,
    ) -> None:
        expected = (self.dim, self.dim)
        field_shapes = {
            "transition_matrix": model_parameters.transition_matrix.shape,
            "transition_noise_cholesky": model_parameters.transition_noise_cholesky.shape,
            "emission_matrix": model_parameters.emission_matrix.shape,
            "emission_noise_cholesky": model_parameters.emission_noise_cholesky.shape,
        }
        for field_name, shape in field_shapes.items():
            if shape != expected:
                raise ValueError(
                    f"{field_name} must have shape {expected} for dim={self.dim}, got {shape}"
                )

    def _validate_inference_parameters(
        self,
        inference_parameters: _UncLGSSMParametersBase,
    ) -> None:
        expected_lower_shape = (_strictly_lower_size(self.dim),)
        expected_shapes = {
            "unc_transition_diag": (self.dim,),
            "emission_strictly_lower": expected_lower_shape,
            "emission_log_diag": (self.dim,),
            "sft_inv_emission_noise_scale": (self.dim,),
        }
        field_values = {
            "unc_transition_diag": inference_parameters.unc_transition_diag,
            "emission_strictly_lower": inference_parameters.emission_strictly_lower,
            "emission_log_diag": inference_parameters.emission_log_diag,
            "sft_inv_emission_noise_scale": inference_parameters.sft_inv_emission_noise_scale,
        }
        for field_name, expected_shape in expected_shapes.items():
            actual_shape = field_values[field_name].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{field_name} must have shape {expected_shape} for dim={self.dim}, got {actual_shape}"
                )

    def to_model_parameters(
        self,
        inference_parameters: _UncLGSSMParametersBase,
    ) -> lg_module._LGSSMParametersBase:
        self._validate_inference_parameters(inference_parameters)
        dtype = inference_parameters.unc_transition_diag.dtype
        parameter_cls = lg_module.make_lgssm_parameters_cls(self.dim)

        transition_diag = jnp.tanh(inference_parameters.unc_transition_diag)
        transition_matrix = jnp.diag(transition_diag)
        transition_noise_cholesky = jnp.eye(self.dim, dtype=dtype)

        emission_matrix = _fill_strictly_lower_triangular(
            inference_parameters.emission_strictly_lower,
            self.dim,
        )
        emission_matrix = emission_matrix.at[jnp.diag_indices(self.dim)].set(
            jnp.exp(inference_parameters.emission_log_diag),
        )

        emission_noise_scale = jax.nn.softplus(
            inference_parameters.sft_inv_emission_noise_scale,
        )
        emission_noise_cholesky = _diag_cholesky(emission_noise_scale)

        return parameter_cls(
            transition_matrix=transition_matrix,
            transition_noise_cholesky=transition_noise_cholesky,
            emission_matrix=emission_matrix,
            emission_noise_cholesky=emission_noise_cholesky,
        )

    def from_model_parameters(
        self,
        model_parameters: lg_module._LGSSMParametersBase,
    ) -> _UncLGSSMParametersBase:
        self._validate_model_parameters(model_parameters)
        transition_diag = jnp.diag(model_parameters.transition_matrix)
        unc_transition_diag = jnp.arctanh(
            jnp.clip(transition_diag, -0.999, 0.999),
        )

        emission_matrix = model_parameters.emission_matrix
        row_ix, col_ix = jnp.tril_indices(self.dim, k=-1)
        emission_strictly_lower = emission_matrix[row_ix, col_ix]

        emission_diag = jnp.diag(emission_matrix)
        if jnp.any(emission_diag <= 0):
            raise ValueError(
                "emission_matrix diagonal must be strictly positive for this parameterization."
            )
        emission_log_diag = jnp.log(emission_diag)

        emission_noise_diag = jnp.diag(model_parameters.emission_noise_cholesky)
        if jnp.any(emission_noise_diag <= 0):
            raise ValueError(
                "emission_noise_cholesky diagonal must be strictly positive."
            )

        return self.inference_parameter_cls(
            unc_transition_diag=unc_transition_diag,
            emission_strictly_lower=emission_strictly_lower,
            emission_log_diag=emission_log_diag,
            sft_inv_emission_noise_scale=_inverse_softplus(emission_noise_diag),
        )

    def sample(self, key: PRNGKeyArray) -> _UncLGSSMParametersBase:
        a_key, l_key, d_key, r_key = jrandom.split(key, 4)
        lower_size = _strictly_lower_size(self.dim)

        unc_transition_diag = 0.3 * jrandom.normal(a_key, shape=(self.dim,))
        emission_strictly_lower = 0.3 * jrandom.normal(l_key, shape=(lower_size,))
        emission_log_diag = 0.2 * jrandom.normal(d_key, shape=(self.dim,))
        emission_noise_scale = jnp.exp(
            -1.0 + 0.5 * jrandom.normal(r_key, shape=(self.dim,))
        )

        return self.inference_parameter_cls(
            unc_transition_diag=unc_transition_diag,
            emission_strictly_lower=emission_strictly_lower,
            emission_log_diag=emission_log_diag,
            sft_inv_emission_noise_scale=_inverse_softplus(emission_noise_scale),
        )

    def log_prob(
        self,
        inference_parameters: _UncLGSSMParametersBase,
    ) -> Scalar:
        self._validate_inference_parameters(inference_parameters)
        emission_noise_scale = jax.nn.softplus(
            inference_parameters.sft_inv_emission_noise_scale,
        )

        transition_logp = jstats.norm.logpdf(
            inference_parameters.unc_transition_diag,
            loc=0.0,
            scale=0.5,
        ).sum()
        emission_lower_logp = jstats.norm.logpdf(
            inference_parameters.emission_strictly_lower,
            loc=0.0,
            scale=0.5,
        ).sum()
        emission_diag_logp = jstats.norm.logpdf(
            inference_parameters.emission_log_diag,
            loc=0.0,
            scale=0.5,
        ).sum()
        scale_logp = (
            jstats.norm.logpdf(
                emission_noise_scale,
                loc=0.0,
                scale=0.5,
            ).sum()
            + emission_noise_scale.shape[0] * jnp.log(jnp.array(2.0))
        )
        jac_logp = jax.nn.log_sigmoid(
            inference_parameters.sft_inv_emission_noise_scale,
        ).sum()

        return (
            transition_logp
            + emission_lower_logp
            + emission_diag_logp
            + scale_logp
            + jac_logp
        )


@jax.tree_util.register_dataclass
@dataclass
class LGSSMBayesian:
    target: interface.SequentialModelProtocol[
        lg_module._VectorStateBase,
        lg_module._VectorObservationBase,
        seqjtyping.NoCondition,
        lg_module._LGSSMParametersBase,
    ]
    parameterization: FullParameterization


def lgssm_full(
    hyperparameters: LGSSMHyperParameters | None = None,
) -> LGSSMBayesian:
    """Factory for identified Bayesian LGSSMs of arbitrary dimension."""
    if hyperparameters is None:
        hyperparameters = LGSSMHyperParameters()

    return LGSSMBayesian(
        target=lg_module.lgssm(hyperparameters.dim),
        parameterization=FullParameterization(hyperparameters=hyperparameters),
    )
