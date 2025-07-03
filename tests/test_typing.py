"""Tests for runtime typing enforcement in model base classes."""

from typing import ClassVar

import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray, Scalar
import equinox as eqx

from seqjax import Emission, Prior, Transition
from seqjax.model.typing import Condition, Observation, Parameters, Particle


class DummyParticle(Particle):
    value: Scalar = eqx.field(default_factory=lambda: jnp.array(0.0))


class DummyObservation(Observation):
    value: Scalar = eqx.field(default_factory=lambda: jnp.array(0.0))


class DummyCondition(Condition):
    value: Scalar = eqx.field(default_factory=lambda: jnp.array(0.0))


class DummyParameters(Parameters):
    reference_emission: tuple[DummyObservation] = eqx.field(
        default_factory=lambda: (DummyObservation(jnp.array(0.0)),)
    )


class MultiParticle(Particle):
    x: Scalar = eqx.field(default_factory=lambda: jnp.array(0.0))
    y: Scalar = eqx.field(default_factory=lambda: jnp.array(0.0))


class MultiObservation(Observation):
    x: Scalar = eqx.field(default_factory=lambda: jnp.array(0.0))
    y: Scalar = eqx.field(default_factory=lambda: jnp.array(0.0))


class GoodPrior(Prior[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[DummyCondition],
        parameters: DummyParameters,
    ) -> tuple[DummyParticle]:
        return (DummyParticle(jnp.array(0.0)),)

    @staticmethod
    def log_prob(
        particle: tuple[DummyParticle],
        conditions: tuple[DummyCondition],
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class GoodTransition(Transition[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[DummyParticle],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyParticle:
        return DummyParticle(jnp.array(0.0))

    @staticmethod
    def log_prob(
        particle_history: tuple[DummyParticle],
        particle: DummyParticle,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class GoodEmission(
    Emission[DummyParticle, DummyObservation, DummyCondition, DummyParameters],
):
    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[DummyParticle],
        observation_history: tuple[DummyObservation],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyObservation:
        return DummyObservation(jnp.array(0.0))

    @staticmethod
    def log_prob(
        particle: tuple[DummyParticle],
        observation_history: tuple[DummyObservation],
        observation: DummyObservation,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


def test_good_classes_instantiation() -> None:
    """Ensure correct subclasses instantiate without error."""

    GoodPrior()
    GoodTransition()
    GoodEmission()


def test_prior_missing_staticmethod() -> None:
    """A non-static method should raise a ``TypeError``."""

    with pytest.raises(TypeError):

        class BadPrior(Prior[DummyParticle, DummyCondition, DummyParameters]):
            order: ClassVar[int] = 1

            def sample(
                key: PRNGKeyArray,
                conditions: tuple[DummyCondition],
                parameters: DummyParameters,
            ) -> tuple[DummyParticle]:  # type: ignore[override]
                return (DummyParticle(jnp.array(0.0)),)

            @staticmethod
            def log_prob(
                particle: tuple[DummyParticle],
                conditions: tuple[DummyCondition],
                parameters: DummyParameters,
            ) -> Scalar:
                return jnp.array(0.0)


def test_as_array_helpers() -> None:
    """``Particle.as_array`` and ``Observation.as_array`` stack leaf values."""

    p = MultiParticle(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    o = MultiObservation(jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0]))

    expected_p = jnp.dstack(
        [jnp.expand_dims(p.x, -1), jnp.expand_dims(p.y, -1)]
    )
    expected_o = jnp.dstack(
        [jnp.expand_dims(o.x, -1), jnp.expand_dims(o.y, -1)]
    )

    assert jnp.array_equal(p.as_array(), expected_p)
    assert jnp.array_equal(o.as_array(), expected_o)


def test_prior_order_mismatch() -> None:
    """Tuple lengths must match ``order``."""

    with pytest.raises(TypeError):

        class BadPrior(Prior[DummyParticle, DummyCondition, DummyParameters]):
            order: ClassVar[int] = 1

            @staticmethod
            def sample(
                key: PRNGKeyArray,
                conditions: tuple[DummyCondition],
                parameters: DummyParameters,
            ) -> tuple[DummyParticle, DummyParticle]:
                return (DummyParticle(jnp.array(0.0)), DummyParticle(jnp.array(0.0)))

            @staticmethod
            def log_prob(
                particle: tuple[DummyParticle, DummyParticle],
                conditions: tuple[DummyCondition],
                parameters: DummyParameters,
            ) -> Scalar:
                return jnp.array(0.0)


def test_transition_order_mismatch() -> None:
    """Tuple lengths must reflect ``order`` in ``Transition``."""

    with pytest.raises(TypeError):

        class BadTransition(Transition[DummyParticle, DummyCondition, DummyParameters]):
            order: ClassVar[int] = 2

            @staticmethod
            def sample(
                key: PRNGKeyArray,
                particle_history: tuple[DummyParticle],
                condition: DummyCondition,
                parameters: DummyParameters,
            ) -> DummyParticle:
                return DummyParticle(jnp.array(0.0))

            @staticmethod
            def log_prob(
                particle_history: tuple[DummyParticle],
                particle: DummyParticle,
                condition: DummyCondition,
                parameters: DummyParameters,
            ) -> Scalar:
                return jnp.array(0.0)


def test_emission_observation_dependency_mismatch() -> None:
    """Observation history length must match ``observation_dependency``."""

    with pytest.raises(TypeError):

        class BadEmission(
            Emission[
                DummyParticle,
                DummyObservation,
                DummyCondition,
                DummyParameters,
            ],
        ):
            order: ClassVar[int] = 1
            observation_dependency: ClassVar[int] = 2

            @staticmethod
            def sample(
                key: PRNGKeyArray,
                particle: tuple[DummyParticle],
                observation_history: tuple[DummyObservation],
                condition: DummyCondition,
                parameters: DummyParameters,
            ) -> DummyObservation:
                return DummyObservation(jnp.array(0.0))

            @staticmethod
            def log_prob(
                particle: tuple[DummyParticle],
                observation_history: tuple[DummyObservation],
                observation: DummyObservation,
                condition: DummyCondition,
                parameters: DummyParameters,
            ) -> Scalar:
                return jnp.array(0.0)
