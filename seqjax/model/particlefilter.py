from typing import NamedTuple, TypeVar, Protocol, Generic, Optional, Callable
from jaxtyping import Array, Float, PRNGKeyArray
from seqjax.target.base import Target, Particle, Observation, Condition, Hyperparameters]

Scalar = Float[Array, ""]

class ParticleFilter(Protocol, Generic[Target]):
    Target[Particle, Observation, Condition, Hyperparameters]