"""Tests for runtime typing enforcement in model base classes."""

from typing import ClassVar

import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray, Scalar
import equinox as eqx

from seqjax.model.base import Prior, Transition, Emission
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
    def log_p(
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
    def log_p(
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
    def log_p(
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
            def log_p(
                particle: tuple[DummyParticle],
                conditions: tuple[DummyCondition],
                parameters: DummyParameters,
            ) -> Scalar:
                return jnp.array(0.0)


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
            def log_p(
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
            def log_p(
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
            def log_p(
                particle: tuple[DummyParticle],
                observation_history: tuple[DummyObservation],
                observation: DummyObservation,
                condition: DummyCondition,
                parameters: DummyParameters,
            ) -> Scalar:
                return jnp.array(0.0)
