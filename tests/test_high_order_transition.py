# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import jax.random as jrandom

from seqjax import simulate
from seqjax.model.base import Prior, Transition, Emission, SequentialModel
from tests.test_typing import (
    DummyParticle,
    DummyObservation,
    DummyCondition,
    DummyParameters,
)
from typing import ClassVar
from jaxtyping import PRNGKeyArray, Scalar


class FibPrior(Prior[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 2

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[DummyCondition, DummyCondition],
        parameters: DummyParameters,
    ) -> tuple[DummyParticle, DummyParticle]:
        _ = key, conditions, parameters
        return DummyParticle(jnp.array(0.0)), DummyParticle(jnp.array(1.0))

    @staticmethod
    def log_prob(
        particle: tuple[DummyParticle, DummyParticle],
        conditions: tuple[DummyCondition, DummyCondition],
        parameters: DummyParameters,
    ) -> Scalar:
        _ = particle, conditions, parameters
        return jnp.array(0.0)


class FibTransition(Transition[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 2

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[DummyParticle, DummyParticle],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyParticle:
        prev_prev, prev = particle_history
        _ = key, condition, parameters
        return DummyParticle(prev_prev.value + prev.value)

    @staticmethod
    def log_prob(
        particle_history: tuple[DummyParticle, DummyParticle],
        particle: DummyParticle,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        _ = particle_history, particle, condition, parameters
        return jnp.array(0.0)


class TrivialEmission(
    Emission[DummyParticle, DummyObservation, DummyCondition, DummyParameters]
):
    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[DummyParticle],
        observation_history: tuple[()],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyObservation:
        _ = key, observation_history, condition, parameters
        return DummyObservation(particle[0].value)

    @staticmethod
    def log_prob(
        particle: tuple[DummyParticle],
        observation_history: tuple[()],
        observation: DummyObservation,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        _ = particle, observation_history, condition, parameters
        return jnp.array(0.0)


class FibModel(
    SequentialModel[DummyParticle, DummyObservation, DummyCondition, DummyParameters]
):
    prior = FibPrior()
    transition = FibTransition()
    emission = TrivialEmission()


def test_simulate_second_order_transition() -> None:
    key = jrandom.PRNGKey(0)
    seq_len = 4
    params = DummyParameters(reference_emission=())
    condition = DummyCondition(jnp.ones(seq_len + FibModel.prior.order - 1))
    latents, _, _, _ = simulate.simulate(
        key, FibModel, condition, params, sequence_length=seq_len
    )
    expected = jnp.array([1.0, 1.0, 2.0, 3.0])
    assert jnp.allclose(latents.value, expected)
