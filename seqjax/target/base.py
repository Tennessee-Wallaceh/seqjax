from typing import NamedTuple, TypeVar, Protocol, Generic, Optional, Callable
from jaxtyping import Array, Float, PRNGKeyArray

# some mildly complicated typing, allows type checker to do nice things.
# use NamedTuples to give variables nice names, working natively with JAX.
# define all in scalar terms, then use jax for array ops.
# use Target class to group together a number of pure functions (staticmethod).
# use contravariant because hyperparams are only an input
# Condition and Hyperparameters are seperated, the idea being Hypers are static across time
# condition is varying.

# Sometimes we need to remap the previous emission into the next condition.
# In these cases we can implement emission_to_condition. The slight wrinkle is that not all
# condition info can come from previous emission, so we also need to pass a partially filled in condition
# and a reference_emission for the first calculation.
# So the abstraction is not perfect, but ok for now.

Particle = TypeVar("Particle", bound=NamedTuple)
Observation = TypeVar("Observation", bound=NamedTuple)
Condition = TypeVar("Condition", bound=NamedTuple)
Hyperparameters = TypeVar("Hyperparameters", bound=NamedTuple, contravariant=True)

Scalar = Float[Array, ""]


class Target(Protocol, Generic[Particle, Observation, Condition, Hyperparameters]):
    @staticmethod
    def sample_prior(
        key: PRNGKeyArray, hyperparameters: Hyperparameters
    ) -> Particle: ...

    @staticmethod
    def prior_log_p(
        particle: Particle, hyperparameters: Hyperparameters
    ) -> Float[Array, ""]: ...

    @staticmethod
    def sample_transition(
        key: PRNGKeyArray,
        particle: Particle,
        condition: Condition,
        hyperparameters: Hyperparameters,
    ) -> Particle: ...

    @staticmethod
    def transition_log_p(
        particle: Particle,
        next_particle: Particle,
        condition: Condition,
        hyperparameters: Hyperparameters,
    ) -> Scalar: ...

    @staticmethod
    def sample_emission(
        key: PRNGKeyArray,
        particle: Particle,
        condition: Condition,
        hyperparameters: Hyperparameters,
    ) -> Observation: ...

    @staticmethod
    def emission_log_p(
        particle: Particle,
        observation: Observation,
        condition: Condition,
        hyperparameters: Hyperparameters,
    ) -> Scalar: ...

    # maps Observation at t-1 and Condition at t to a new Condition at t
    @staticmethod
    def emission_to_condition(
        observation: Observation,
        condition: Condition,
        hyperparameters: Hyperparameters,
    ) -> Condition:
        return condition

    @staticmethod
    def reference_emission(
        hyperparameters: Hyperparameters,
    ) -> Optional[Observation]:
        return None
