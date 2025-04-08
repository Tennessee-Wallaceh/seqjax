from typing import TypeVar, Protocol, Generic, Optional

from flax import struct
from jaxtyping import PRNGKeyArray, Scalar, Float, Array
from typing import Type, Any, get_type_hints

# use flax.struct to give variables nice names, working with JAX and giving a level of typing.
# define all in scalar terms, then use jax vmapping for array ops.
# e.g Particle and Observation
# Condition and Parameters are seperated, the idea being Hypers are static across time
# condition is varying.

# use Target class to group together a number of pure functions (staticmethod), that operate on the same
# Particle, Observation etc.
# Prior, Transition, Emission give additional levels of grouping, since in practice these will always be paired.

# Sometimes we need to remap the previous emission into the next condition.
# In these cases we can implement emission_to_condition. The slight wrinkle is that not all
# condition info can come from previous emission, so we also need to pass a partially filled in condition
# and a reference_emission for the first calculation.
# So the abstraction is not perfect, but ok for now.


class Particle(struct.PyTreeNode): ...


class Observation(struct.PyTreeNode): ...


class Condition(struct.PyTreeNode): ...


class Parameters(struct.PyTreeNode): ...


ParticleType = TypeVar("ParticleType", bound=Particle)
ObservationType = TypeVar("ObservationType", bound=Observation)
ConditionType = TypeVar("ConditionType", bound=Condition)
ParametersType = TypeVar("ParametersType", bound=Parameters)


def vectorize_type(base_type: Type[Any], **dims) -> Type[Any]:
    # Create a new type with modified field annotations
    annotations = get_type_hints(base_type)
    new_annotations = {
        field: Float[Array, *dims.get(field, (...,))]  # Add dimensions dynamically
        for field, t in annotations.items()
    }

    # Use `type` to dynamically create a new class type
    return type(
        f"Vectorized{base_type.__name__}",
        (base_type,),
        {"__annotations__": new_annotations},
    )


class Prior(Protocol, Generic[ParticleType, ParametersType]):
    @staticmethod
    def sample(key: PRNGKeyArray, parameters: ParametersType) -> ParticleType: ...

    @staticmethod
    def log_p(particle: ParticleType, parameters: ParametersType) -> Scalar: ...


class Transition(Protocol, Generic[ParticleType, ConditionType, ParametersType]):
    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: ParticleType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> ParticleType:
        raise NotImplementedError

    @staticmethod
    def log_p(
        particle: ParticleType,
        next_particle: ParticleType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar:
        raise NotImplementedError


class Emission(
    Protocol, Generic[ParticleType, ConditionType, ObservationType, ParametersType]
):
    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: ParticleType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> ObservationType:
        raise NotImplementedError

    @staticmethod
    def log_p(
        particle: ParticleType,
        observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar:
        raise NotImplementedError


class Target(
    Protocol, Generic[ParticleType, ConditionType, ObservationType, ParametersType]
):
    prior: Prior[ParticleType, ParametersType]
    transition: Transition[ParticleType, ConditionType, ParametersType]
    emission: Emission[ParticleType, ConditionType, ObservationType, ParametersType]

    # maps Observation at t-1 and Condition at t to a new Condition at t
    @staticmethod
    def emission_to_condition(
        observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> ConditionType:
        return condition

    # need to define the initial reference emission
    # TODO: should this be part of the prior?
    @staticmethod
    def reference_emission(
        parameters: ParametersType,
    ) -> Optional[ObservationType]:
        return None
