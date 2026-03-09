"""Bayesian parameterisation for the linear Gaussian state-space model."""

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


class UncLGSSMParameters(seqjtyping.Parameters):
    """Unconstrained parameterisation of LGSSM parameters."""

    transition_matrix: jax.Array = field(default_factory=lambda: jnp.eye(50))
    sft_inv_transition_noise_scale: jax.Array = field(default_factory=lambda: jnp.zeros(50))
    emission_matrix: jax.Array = field(default_factory=lambda: jnp.eye(50))
    sft_inv_emission_noise_scale: jax.Array = field(default_factory=lambda: jnp.zeros(50))

    _shape_template: typing.ClassVar = OrderedDict(
        transition_matrix=jax.ShapeDtypeStruct(shape=(50, 50), dtype=jnp.float32),
        sft_inv_transition_noise_scale=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
        emission_matrix=jax.ShapeDtypeStruct(shape=(50, 50), dtype=jnp.float32),
        sft_inv_emission_noise_scale=jax.ShapeDtypeStruct(shape=(50,), dtype=jnp.float32),
    )


@dataclass
class FullParameterization(
    ParameterizationProtocol[
        lg_module.LGSSMParameters,
        UncLGSSMParameters,
        seqjtyping.NoHyper,
    ]
):
    """Infer transition/emission matrices and diagonal noise scales."""

    inference_parameter_cls: type[UncLGSSMParameters] = UncLGSSMParameters
    hyperparameters: seqjtyping.NoHyper = field(default_factory=seqjtyping.NoHyper)

    def to_model_parameters(
        self,
        inference_parameters: UncLGSSMParameters,
    ) -> lg_module.LGSSMParameters:
        transition_noise_scale = jax.nn.softplus(
            inference_parameters.sft_inv_transition_noise_scale,
        )
        emission_noise_scale = jax.nn.softplus(
            inference_parameters.sft_inv_emission_noise_scale,
        )

        dim = transition_noise_scale.shape[-1]
        eye = jnp.eye(dim, dtype=transition_noise_scale.dtype)

        return lg_module.LGSSMParameters(
            transition_matrix=inference_parameters.transition_matrix,
            transition_noise_scale=transition_noise_scale,
            transition_noise_corr_cholesky=eye,
            emission_matrix=inference_parameters.emission_matrix,
            emission_noise_scale=emission_noise_scale,
            emission_noise_corr_cholesky=eye,
        )

    def from_model_parameters(
        self,
        model_parameters: lg_module.LGSSMParameters,
    ) -> UncLGSSMParameters:
        return UncLGSSMParameters(
            transition_matrix=model_parameters.transition_matrix,
            sft_inv_transition_noise_scale=_inverse_softplus(
                model_parameters.transition_noise_scale,
            ),
            emission_matrix=model_parameters.emission_matrix,
            sft_inv_emission_noise_scale=_inverse_softplus(
                model_parameters.emission_noise_scale,
            ),
        )

    def sample(self, key: PRNGKeyArray) -> UncLGSSMParameters:
        a_key, q_key, c_key, r_key = jrandom.split(key, 4)

        transition_matrix = jrandom.normal(a_key, shape=(50, 50))
        emission_matrix = jrandom.normal(c_key, shape=(50, 50))

        transition_noise_scale = jnp.abs(jrandom.cauchy(q_key, shape=(50,)))
        emission_noise_scale = jnp.abs(jrandom.cauchy(r_key, shape=(50,)))

        return UncLGSSMParameters(
            transition_matrix=transition_matrix,
            sft_inv_transition_noise_scale=_inverse_softplus(transition_noise_scale),
            emission_matrix=emission_matrix,
            sft_inv_emission_noise_scale=_inverse_softplus(emission_noise_scale),
        )

    def log_prob(
        self,
        inference_parameters: UncLGSSMParameters,
    ) -> Scalar:
        transition_noise_scale = jax.nn.softplus(
            inference_parameters.sft_inv_transition_noise_scale,
        )
        emission_noise_scale = jax.nn.softplus(
            inference_parameters.sft_inv_emission_noise_scale,
        )

        matrix_logp = jstats.norm.logpdf(
            inference_parameters.transition_matrix,
            loc=0.0,
            scale=1.0,
        ).sum() + jstats.norm.logpdf(
            inference_parameters.emission_matrix,
            loc=0.0,
            scale=1.0,
        ).sum()

        scale_logp = (
            jstats.cauchy.logpdf(transition_noise_scale).sum()
            + jstats.cauchy.logpdf(emission_noise_scale).sum()
            + 50 * jnp.log(jnp.array(2.0))
            + 50 * jnp.log(jnp.array(2.0))
        )

        lad = jax.nn.log_sigmoid(
            inference_parameters.sft_inv_transition_noise_scale,
        ).sum() + jax.nn.log_sigmoid(
            inference_parameters.sft_inv_emission_noise_scale,
        ).sum()

        return matrix_logp + scale_logp + lad


@dataclass
class LGSSMBayesian:
    target: typing.ClassVar[
        interface.SequentialModelProtocol[
            lg_module.VectorState,
            lg_module.VectorObservation,
            seqjtyping.NoCondition,
            lg_module.LGSSMParameters,
        ]
    ] = typing.cast(interface.SequentialModelProtocol, lg_module)
    parameterization: FullParameterization


def lgssm_full(hyperparameters: typing.Any) -> LGSSMBayesian:
    """Factory for full-parameter Bayesian LGSSM."""

    _ = hyperparameters
    return LGSSMBayesian(parameterization=FullParameterization())
