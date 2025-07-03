import pytest

# mark requires jax
jax = pytest.importorskip("jax")
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import simulate
from seqjax.util import pytree_shape
from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model.stochastic_vol import SimpleStochasticVol, LogVolRW, TimeIncrement
from seqjax.model.base import Prior, Transition, Emission, SequentialModel
from tests.test_typing import (
    DummyParticle,
    DummyObservation,
    DummyCondition,
    DummyParameters,
)
from typing import ClassVar
from jaxtyping import PRNGKeyArray, Scalar


def test_ar1_target_simulate_length() -> None:
    key = jax.random.PRNGKey(0)
    params = ARParameters()
    latent, obs = simulate.simulate(key, AR1Target, None, params, sequence_length=3)

    assert latent.x.shape == (3,)
    assert obs.y.shape == (3,)


def test_simple_stochastic_vol_simulate_length() -> None:
    key = jax.random.PRNGKey(0)
    params = LogVolRW(
        std_log_vol=jnp.array(0.1),
        mean_reversion=jnp.array(0.1),
        long_term_vol=jnp.array(1.0),
    )
    cond = TimeIncrement(jnp.array([1.0, 1.0, 1.0, 1.0]))
    latent, obs = simulate.simulate(
        key, SimpleStochasticVol, cond, params, sequence_length=3
    )

    assert latent.log_vol.shape == (3,)
    assert obs.underlying.shape == (3,)


class Prior1(Prior[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[DummyCondition],
        parameters: DummyParameters,
    ) -> tuple[DummyParticle]:
        return (DummyParticle(jrandom.normal(key)),)

    @staticmethod
    def log_p(
        particle: tuple[DummyParticle],
        conditions: tuple[DummyCondition],
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Transition1(Transition[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[DummyParticle],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyParticle:
        return DummyParticle(jrandom.normal(key))

    @staticmethod
    def log_p(
        particle_history: tuple[DummyParticle],
        particle: DummyParticle,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Emission1(Emission[DummyParticle, DummyObservation, DummyCondition, DummyParameters]):
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
        return DummyObservation(jrandom.normal(key))

    @staticmethod
    def log_p(
        particle: tuple[DummyParticle],
        observation_history: tuple[()],
        observation: DummyObservation,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Target1(SequentialModel[DummyParticle, DummyObservation, DummyCondition, DummyParameters]):
    prior = Prior1()
    transition = Transition1()
    emission = Emission1()


class Prior2(Prior[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 2

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[DummyCondition, DummyCondition],
        parameters: DummyParameters,
    ) -> tuple[DummyParticle, DummyParticle]:
        k1, k2 = jrandom.split(key)
        return (
            DummyParticle(jrandom.normal(k1)),
            DummyParticle(jrandom.normal(k2)),
        )

    @staticmethod
    def log_p(
        particle: tuple[DummyParticle, DummyParticle],
        conditions: tuple[DummyCondition, DummyCondition],
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Transition2(Transition[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[DummyParticle],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyParticle:
        return DummyParticle(jrandom.normal(key))

    @staticmethod
    def log_p(
        particle_history: tuple[DummyParticle],
        particle: DummyParticle,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Emission2(Emission[DummyParticle, DummyObservation, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 2
    observation_dependency: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[DummyParticle, DummyParticle],
        observation_history: tuple[DummyObservation],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyObservation:
        return DummyObservation(jrandom.normal(key))

    @staticmethod
    def log_p(
        particle: tuple[DummyParticle, DummyParticle],
        observation_history: tuple[DummyObservation],
        observation: DummyObservation,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Target2(SequentialModel[DummyParticle, DummyObservation, DummyCondition, DummyParameters]):
    prior = Prior2()
    transition = Transition2()
    emission = Emission2()


@pytest.mark.parametrize(
    "target,obs_dep,seq_len,cond_len,should_fail",
    [
        (Target1, 0, 3, 3, False),
        (Target2, 1, 3, 4, False),
        (Target2, 1, 3, 3, True),
    ],
)
def test_simulate_dependency_lengths(
    target,
    obs_dep,
    seq_len,
    cond_len,
    should_fail,
) -> None:
    key = jax.random.PRNGKey(1)
    params = DummyParameters(
        reference_emission=tuple(
            DummyObservation(jnp.array(0.0)) for _ in range(obs_dep)
        )
    )
    condition = DummyCondition(jnp.ones(cond_len))
    if should_fail:
        with pytest.raises(jax.errors.JaxRuntimeError):
            simulate.simulate(key, target, condition, params, seq_len)
    else:
        latent, obs = simulate.simulate(key, target, condition, params, seq_len)
        assert pytree_shape(latent)[0][0] == seq_len
        assert pytree_shape(obs)[0][0] == seq_len
