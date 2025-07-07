from abc import abstractmethod

import equinox as eqx

import jax
import jax.random as jrandom

from seqjax.model.base import ParticleType, ParametersType
from seqjax.model.typing import (
    Batched,
    SequenceAxis,
    SampleAxis,
    ParamAxis,
    HyperParametersType,
)
from typing import Generic, TypeVar

from jaxtyping import Shaped, Array, Int, Float, PRNGKeyArray

ContextAxis = TypeVar("ContextAxis", covariant=True)
SamplePathAxis = TypeVar("SamplePathAxis", covariant=True)


class VariationalLatentPosterior(eqx.Module, Generic[ParticleType, ParametersType]):
    # this indicates that the AmortizedSampler are equinox Modules (dataclass pytrees)
    # and implement a sample_single_path method with the following interface
    # from this we can define batched sampling operations
    # the idea is that these samplers operate as functions of some context and
    # a parameter set
    sample_length: int  # length of batches
    x_dim: int  # dimension of sampled x
    context_dim: int
    parameter_dim: int

    @abstractmethod
    def sample_single_path(
        self,
        key: PRNGKeyArray,
        p_context: ParametersType,
        y_context: Batched[ParticleType, SamplePathAxis],
    ) -> tuple[Batched[ParticleType, SamplePathAxis], Batched[Float, SamplePathAxis]]:
        pass

    def sample_for_context(
        self,
        key: PRNGKeyArray,
        p_context: Batched[ParticleType, ContextAxis],
        y_context: Batched[ParticleType, ContextAxis, SamplePathAxis],
        samples_per_context: int,
    ) -> tuple[
        Float[Array, "samples_per_context sample_length x_dim"],
        Float[Array, "samples_per_context sample_length"],
        Batched[ParticleType, SampleAxis, SamplePathAxis],
    ]:
        # leading axis of theta is the number of samples per context
        # so for this context we sample matching the theta leading axis
        keys = jrandom.split(key, samples_per_context)
        x_approx, log_q_x = jax.vmap(self.sample_single_path, in_axes=[0, 0, None])(
            keys, theta_context, context
        )
        return x_approx, log_q_x


class VariationalJoint(eqx.Module, Generic[ParticleType, ParametersType]):
    variational_latent: VariationalLatentPosterior[ParticleType, ParametersType]
    variational_parameter: VariationalParameterPosterior[ParametersType]

    # model knows about the target to automatically build structs
    target_particle: ParticleType

    # accept an array of keys corresponding to data
    # sharding of keys should match leading axis of y_observations
    def sample_and_log_prob(
        self,
        y_observations: Any,
        keys: Shaped[PRNGKeyArray, "num_context"],
        samples_per_context: int,
    ) -> tuple[
        Float[Array, "num_context samples_per_context sample_length x_dim"],  # x_approx
        Float[Array, "num_context samples_per_context"],  # log_q_x
        Float[Array, "num_context samples_per_context parameter_dim"],  # theta_approx
        Float[Array, "num_context samples_per_context"],  # log_q_theta
    ]:
        split_k = jax.vmap(jrandom.split)(keys)
        theta_keys, x_keys = split_k[:, 0], split_k[:, 1]
        context = jax.vmap(self.embedder.embed)(
            y_observations.as_array()
        )  # ["num_context sample_length context_dim"]
        theta_array, log_q_theta = jax.vmap(
            self.parameter_model.sample_array_and_log_prob, in_axes=[0, None]
        )(theta_keys, samples_per_context)
        x, log_q_x = jax.vmap(self.sampler.sample_for_context, in_axes=[0, 0, 0, None])(
            x_keys, jax.lax.stop_gradient(theta_array), context, samples_per_context
        )

        # convert to struct
        x_struct = self.target_particle.from_array(x)
        theta_struct = self.parameter_model.array_to_struct(theta_array)
        theta_struct = broadcast_pytree(theta_struct, infer_pytree_shape(theta_struct))

        return x_struct, log_q_x, theta_struct, log_q_theta

    # accept an array of keys corresponding to data
    # sharding of keys should match leading axis of y_observations
    def sample_theta_and_log_prob(
        self, keys: Shaped[PRNGKeyArray, "num_context"], samples_per_context: int
    ) -> tuple[
        Float[Array, "num_context samples_per_context parameter_dim"],  # theta_approx
        Float[Array, "num_context samples_per_context"],  # log_q_theta
    ]:
        theta_array, log_q_theta = jax.vmap(
            self.parameter_model.sample_array_and_log_prob, in_axes=[0, None]
        )(keys, samples_per_context)
        theta_struct = self.parameter_model.array_to_struct(theta_array)
        return theta_struct, log_q_theta
