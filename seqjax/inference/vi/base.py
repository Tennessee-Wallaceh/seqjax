from abc import abstractmethod

import equinox as eqx

import jax
import jax.random as jrandom

from seqjax.model.base import (
    ObservationType,
    ConditionType,
    ParticleType,
    ParametersType,
)
from seqjax.model.typing import (
    Batched,
    SequenceAxis,
    SampleAxis,
    HyperParametersType,
)
from seqjax.inference.vi.parameter import VariationalParameterPosterior

from typing import Generic, TypeVar

from jaxtyping import Shaped, Array, Int, Float, PRNGKeyArray

ContextAxis = TypeVar("ContextAxis", covariant=True)
SamplePathAxis = TypeVar("SamplePathAxis", covariant=True)


class VariationalLatentPosterior(
    eqx.Module, Generic[ObservationType, ConditionType, ParticleType, ParametersType]
):
    """
    Amortised variational posterior **q(x | y, θ)** implemented as an
    Equinox ``Module`` that *samples entire latent trajectories* rather
    than isolated points.

    The class is intentionally lightweight: it specifies the interface
    (particularly :pymeth:`sample_single_path`) and stores only the
    dimensions needed to sanity-check shapes at run-time.

    ----------
    Attributes
    ----------
    sample_length : int
        Number of steps in each trajectory produced by :pymeth:`sample_single_path`.

    x_dim : int
        Dimensionality of each latent point *zₜ* returned.
    context_dim : int
        Dimensionality of the **conditioning context** *y* that drives
        the amortisation network.
    parameter_dim : int
        Dimensionality of the parameter vector *θ* supplied in
        ``p_context``.

    ----------
    Notes
    -----
    * **Stateless by design.**  All variability comes from the PRNG key
      and the context tensors you pass in; the module’s own parameters
      are fixed once initialised.
    * The class is *generic* over ``ParticleType`` (dtype/structure of a
      single latent particle) and ``ParametersType`` (dtype/structure of
      θ).  This lets you plug in e.g. simple floats, PyTrees, or even
      namedtuples without rewriting the sampler.


    ----------
    Examples
    --------
    >>> key = jax.random.PRNGKey(0)
    >>> sampler = MyPosterior(sample_length=20, x_dim=4,
    ...                       context_dim=3, parameter_dim=8)
    >>> x_path, log_q = sampler.sample_single_path(
    ...         key, θ, y_context)          # shapes: (20, 4), (20,)
    """

    sample_length: int  # length of batches, corresponds to the SamplePathAxis
    x_dim: int  # dimension of sampled x
    context_dim: int
    parameter_dim: int

    @abstractmethod
    def sample_single_path(
        self,
        key: PRNGKeyArray,
        p_context: ParametersType,
        y_context: Batched[ObservationType, SamplePathAxis],
        c_context: Batched[ConditionType, SamplePathAxis],
    ) -> tuple[Batched[ParticleType, SamplePathAxis], Batched[Float, SamplePathAxis]]:
        """ """
        pass


class VariationalJoint(
    eqx.Module, Generic[ObservationType, ConditionType, ParticleType, ParametersType]
):
    variational_latent: VariationalLatentPosterior[
        ObservationType, ConditionType, ParticleType, ParametersType
    ]
    variational_parameter: VariationalParameterPosterior[ParametersType]

    # model knows about the target to automatically build structs
    target_particle: ParticleType

    # accept an array of keys corresponding to data
    # sharding of keys should match leading axis of y_observa
    def sample_and_log_prob(
        self,
        y_context: Batched[ObservationType, ContextAxis, SamplePathAxis],
        c_context: Batched[ConditionType, ContextAxis, SamplePathAxis],
        keys: Batched[PRNGKeyArray, ContextAxis],
        samples_per_context: int,
    ) -> tuple[
        Batched[ParticleType, ContextAxis, SampleAxis, SamplePathAxis],
        Batched[Float, ContextAxis, SampleAxis],
        Float[Array, "num_context samples_per_context sample_length x_dim"],  # x_approx
        Float[Array, "num_context samples_per_context"],  # log_q_x
        Float[Array, "num_context samples_per_context parameter_dim"],  # theta_approx
        Float[Array, "num_context samples_per_context"],  # log_q_theta
    ]:
        split_k = jax.vmap(jrandom.split)(keys)
        theta_keys, x_keys = split_k[:, 0], split_k[:, 1]
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
