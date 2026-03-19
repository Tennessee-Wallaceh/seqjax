"""Bayesian parameterisation for an identified 5D LGSSM."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar


import seqjax.model.typing as seqjtyping
from . import interface, linear_gaussian as lg_module
from .interface import ParameterizationProtocol

def _inverse_softplus(x: jax.Array) -> jax.Array:
    if jnp.any(x <= 0):
        raise ValueError("All scale parameters must be strictly positive")
    return jnp.log(jnp.expm1(x))


def _diag_cholesky(scale: jax.Array) -> jax.Array:
    return jnp.diag(scale)


def _fill_strictly_lower_triangular(vec: jax.Array, dim: int) -> jax.Array:
    """Fill the strictly lower-triangular part of a matrix from a vector."""
    mat = jnp.zeros((dim, dim), dtype=vec.dtype)
    row_ix, col_ix = jnp.tril_indices(dim, k=-1)
    return mat.at[row_ix, col_ix].set(vec)

@dataclass
class UncLGSSMParameters(seqjtyping.Parameters):
    """Unconstrained parameterisation of an identified 5D LGSSM.

    Structure:
    - diagonal stable transition_matrix
    - fixed identity transition covariance
    - lower-triangular emission_matrix with positive diagonal
    - diagonal emission covariance
    """

    # Diagonal AR coefficients, mapped through tanh to (-1, 1)
    unc_transition_diag: jax.Array = field(default_factory=lambda: jnp.zeros(5))

    # Strictly lower-triangular entries of emission matrix
    emission_strictly_lower: jax.Array = field(default_factory=lambda: jnp.zeros(10))

    # Positive diagonal of emission matrix via exp
    emission_log_diag: jax.Array = field(default_factory=lambda: jnp.zeros(5))

    # Diagonal emission noise via softplus
    sft_inv_emission_noise_scale: jax.Array = field(default_factory=lambda: jnp.zeros(5))

    _shape_template: typing.ClassVar = OrderedDict(
        unc_transition_diag=jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32),
        emission_strictly_lower=jax.ShapeDtypeStruct(shape=(10,), dtype=jnp.float32),
        emission_log_diag=jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32),
        sft_inv_emission_noise_scale=jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32),
    )


@jax.tree_util.register_dataclass
@dataclass
class FullParameterization(
    ParameterizationProtocol[
        lg_module.LGSSMParameters,
        UncLGSSMParameters,
        seqjtyping.NoHyper,
    ]
):
    """Identified 5D LGSSM parameterisation.

    Model-space structure:
    - diagonal stable transition_matrix
    - fixed identity transition noise cholesky
    - lower-triangular emission_matrix with positive diagonal
    - diagonal emission noise cholesky
    """

    inference_parameter_cls: type[UncLGSSMParameters] = UncLGSSMParameters
    hyperparameters: seqjtyping.NoHyper = field(default_factory=seqjtyping.NoHyper)

    def to_model_parameters(
        self,
        inference_parameters: UncLGSSMParameters,
    ) -> lg_module.LGSSMParameters:
        dim = inference_parameters.unc_transition_diag.shape[0]
        dtype = inference_parameters.unc_transition_diag.dtype

        # Stable diagonal transition matrix
        transition_diag = jnp.tanh(inference_parameters.unc_transition_diag)
        transition_matrix = jnp.diag(transition_diag)

        # Fixed identity transition noise
        transition_noise_cholesky = jnp.eye(dim, dtype=dtype)

        # Lower-triangular emission matrix with positive diagonal
        emission_matrix = _fill_strictly_lower_triangular(
            inference_parameters.emission_strictly_lower,
            dim,
        )
        emission_matrix = emission_matrix.at[jnp.diag_indices(dim)].set(
            jnp.exp(inference_parameters.emission_log_diag),
        )

        # Diagonal emission noise
        emission_noise_scale = jax.nn.softplus(
            inference_parameters.sft_inv_emission_noise_scale,
        )
        emission_noise_cholesky = _diag_cholesky(emission_noise_scale)

        return lg_module.LGSSMParameters(
            transition_matrix=transition_matrix,
            transition_noise_cholesky=transition_noise_cholesky,
            emission_matrix=emission_matrix,
            emission_noise_cholesky=emission_noise_cholesky,
        )

    def from_model_parameters(
        self,
        model_parameters: lg_module.LGSSMParameters,
    ) -> UncLGSSMParameters:
        transition_diag = jnp.diag(model_parameters.transition_matrix)
        unc_transition_diag = jnp.arctanh(
            jnp.clip(transition_diag, -0.999, 0.999),
        )

        dim = transition_diag.shape[0]

        emission_matrix = model_parameters.emission_matrix
        row_ix, col_ix = jnp.tril_indices(dim, k=-1)
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

        return UncLGSSMParameters(
            unc_transition_diag=unc_transition_diag,
            emission_strictly_lower=emission_strictly_lower,
            emission_log_diag=emission_log_diag,
            sft_inv_emission_noise_scale=_inverse_softplus(emission_noise_diag),
        )

    def sample(self, key: PRNGKeyArray) -> UncLGSSMParameters:
        a_key, l_key, d_key, r_key = jrandom.split(key, 4)


        unc_transition_diag = 0.3 * jrandom.normal(a_key, shape=(5,))

        emission_strictly_lower = 0.3 * jrandom.normal(l_key, shape=(10,))
        emission_log_diag = 0.2 * jrandom.normal(d_key, shape=(5,))

        emission_noise_scale = jnp.exp(
            -1.0 + 0.5 * jrandom.normal(r_key, shape=(5,))
        )

        return UncLGSSMParameters(
            unc_transition_diag=unc_transition_diag,
            emission_strictly_lower=emission_strictly_lower,
            emission_log_diag=emission_log_diag,
            sft_inv_emission_noise_scale=_inverse_softplus(emission_noise_scale),
        )

    def log_prob(
        self,
        inference_parameters: UncLGSSMParameters,
    ) -> Scalar:
        emission_noise_scale = jax.nn.softplus(
            inference_parameters.sft_inv_emission_noise_scale,
        )

        # Prior on unconstrained diagonal AR terms
        transition_logp = jstats.norm.logpdf(
            inference_parameters.unc_transition_diag,
            loc=0.0,
            scale=0.5,
        ).sum()

        # Shrinkage prior on strictly lower-triangular loadings
        emission_lower_logp = jstats.norm.logpdf(
            inference_parameters.emission_strictly_lower,
            loc=0.0,
            scale=0.5,
        ).sum()

        # Prior on log diagonal of emission matrix
        emission_diag_logp = jstats.norm.logpdf(
            inference_parameters.emission_log_diag,
            loc=0.0,
            scale=0.5,
        ).sum()

        # Half-normal prior on emission noise scales, with transform Jacobian
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
    target: typing.ClassVar[
        interface.SequentialModelProtocol[
            lg_module.VectorState,
            lg_module.VectorObservation,
            seqjtyping.NoCondition,
            lg_module.LGSSMParameters,
        ]
    ] = lg_module.lgssm_model
    parameterization: FullParameterization


def lgssm_full(hyperparameters: typing.Any = None) -> LGSSMBayesian:
    """Factory for identified 5D Bayesian LGSSM."""

    _ = hyperparameters
    return LGSSMBayesian(parameterization=FullParameterization())