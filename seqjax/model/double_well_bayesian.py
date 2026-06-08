"""Bayesian parameterisations for the double-well model."""

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
from seqjax.model.interface import ParameterizationProtocol
from . import double_well as dw_model

@dataclass
class UncEBOnlyParameters(seqjtyping.Parameters):
    """Unconstrained energy-barrier parameterisation via softplus inverse."""

    sft_inv_energy_barrier: Scalar = field(default_factory=lambda: jnp.array(0.0))

    _shape_template: typing.ClassVar = OrderedDict(
        sft_inv_energy_barrier=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

@dataclass
class FixedEBParameters(seqjtyping.HyperParameters):
    fixed_observation_std: Scalar = field(default_factory=lambda: jnp.array(0.5))
    fixed_transition_std: Scalar = field(default_factory=lambda: jnp.array(1.0))
    _shape_template: typing.ClassVar = OrderedDict(
        fixed_observation_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        fixed_transition_std=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )

def _energy_barrier_prior_log_prob(energy_barrier: Scalar) -> Scalar:
    """Log-normal prior on the positive energy barrier."""

    return jstats.norm.logpdf(jnp.log(energy_barrier), scale=jnp.array(1.0))


@jax.tree_util.register_dataclass
@dataclass
class EBOnlyParameterization(
    ParameterizationProtocol[
        dw_model.DoubleWellParams,
        UncEBOnlyParameters,
        FixedEBParameters,
    ]
):
    """Infer only energy barrier; observation/transition std are fixed by hyperparameters."""

    inference_parameter_cls: type[UncEBOnlyParameters] = UncEBOnlyParameters
    hyperparameters: dw_model.DoubleWellParams = field(default_factory=FixedEBParameters)

    def to_model_parameters(
        self,
        inference_parameters: UncEBOnlyParameters,
    ) -> dw_model.DoubleWellParams:
        energy_barrier = jax.nn.softplus(inference_parameters.sft_inv_energy_barrier)
        return dw_model.DoubleWellParams(
            energy_barrier=energy_barrier,
            observation_std=jnp.array(self.hyperparameters.fixed_observation_std),
            transition_std=jnp.array(self.hyperparameters.fixed_transition_std),
        )

    def from_model_parameters(
        self,
        model_parameters: dw_model.DoubleWellParams,
    ) -> UncEBOnlyParameters:
        if jnp.any(model_parameters.energy_barrier <= 0):
            raise ValueError(
                "energy_barrier must be strictly positive when converting to "
                "unconstrained parameters"
            )

        return UncEBOnlyParameters(
            sft_inv_energy_barrier=jnp.log(jnp.expm1(model_parameters.energy_barrier)),
        )

    def sample(self, key: PRNGKeyArray) -> UncEBOnlyParameters:
        energy_barrier = jrandom.lognormal(key, sigma=jnp.array(1.0))
        return UncEBOnlyParameters(
            sft_inv_energy_barrier=jnp.log(jnp.expm1(energy_barrier)),
        )

    def log_prob(self, inference_parameters: UncEBOnlyParameters) -> Scalar:
        x = inference_parameters.sft_inv_energy_barrier
        energy_barrier = jax.nn.softplus(x)

        # log abs det jacobian for softplus transform
        lad = jax.nn.log_sigmoid(x)
        return _energy_barrier_prior_log_prob(energy_barrier) + lad

@jax.tree_util.register_dataclass
@dataclass
class DoubleWellBayesian:
    target: typing.ClassVar = dw_model.double_well_model
    parameterization: EBOnlyParameterization


def eb_only(hyperparameters: FixedEBParameters) -> DoubleWellBayesian:
    """Factory for Bayesian double-well model with EB-only inference parameters."""

    return DoubleWellBayesian(
        parameterization=EBOnlyParameterization(hyperparameters=hyperparameters),
    )
