from abc import abstractmethod
import typing
from functools import partial

import equinox as eqx
import jax.scipy.stats as jstats
import jaxtyping

import jax.numpy as jnp
import jax
from jax.nn import softplus
import jax.random as jrandom


import seqjax
import seqjax.model
import seqjax.model.typing
from seqjax import util
from seqjax.inference.embedder import Embedder, NullEmbedder


class VariationalApproximation[
    TargetStructT: seqjax.model.typing.Packable,
    ConditionT,
](eqx.Module):
    target_struct_cls: type[TargetStructT] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)

    @abstractmethod
    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: ConditionT,
    ) -> tuple[
        TargetStructT,
        jaxtyping.Scalar,
    ]: ...


class AmortizedVariationalApproximation[
    TargetStructT: seqjax.model.typing.Packable,
    ConditionT: tuple[seqjax.model.typing.Parameters, seqjax.model.typing.Observation],
](VariationalApproximation[TargetStructT, ConditionT]): ...


class UnconditionalVariationalApproximation[
    TargetStructT: seqjax.model.typing.Packable,
](VariationalApproximation[TargetStructT, None]):
    @abstractmethod
    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: None = None,
    ) -> tuple[TargetStructT, jaxtyping.Scalar]: ...


class VariationalApproximationFactory[
    TargetStructT: seqjax.model.typing.Packable,
    ConditionT,
](typing.Protocol):
    def __call__(
        self, target_struct_cls: type[TargetStructT]
    ) -> VariationalApproximation[TargetStructT, ConditionT]: ...


class MeanField[TargetStructT: seqjax.model.typing.Packable](
    UnconditionalVariationalApproximation[TargetStructT]
):
    target_struct_cls: type[TargetStructT]
    loc: jaxtyping.Float[jaxtyping.Array, "TargetStructT.dim"]
    _unc_scale: jaxtyping.Float[jaxtyping.Array, "TargetStructT.dim"]

    def __init__(self, target_struct_cls: type[TargetStructT]):
        super().__init__(target_struct_cls, shape=(target_struct_cls.flat_dim,))
        self.target_struct_cls = target_struct_cls
        self.loc = jnp.zeros((target_struct_cls.flat_dim,))
        self._unc_scale = jnp.zeros((target_struct_cls.flat_dim,))

    def sample_and_log_prob(self, key, condition=None):
        z = jrandom.normal(key, [self.target_struct_cls.flat_dim])
        scale = 1e-6 + softplus(self._unc_scale)
        x = z * scale + self.loc
        log_q_x = jstats.norm.logpdf(x, loc=self.loc, scale=scale)
        return self.target_struct_cls.unravel(x), jnp.sum(log_q_x)


def buffer_params(
    parameters, buffer_mask, observations_per_step, samples_per_context, sample_length
):
    """
    Apply masking to propogate gradients wrt to theta depending on location in sample.
    """

    def expand_param(p):
        return jnp.broadcast_to(
            p[..., None], (observations_per_step, samples_per_context, sample_length)
        )

    theta_grad = jax.tree.map(expand_param, parameters)
    theta_no_grad = jax.tree.map(expand_param, jax.lax.stop_gradient(parameters))

    # don't take grad when inside the buffer
    return jax.tree.map(
        lambda a, b: jnp.where(buffer_mask, a, b),
        theta_no_grad,
        theta_grad,
    )


class SSMVariationalApproximation[
    LatentStructT: seqjax.model.typing.Particle,
    ParameterStructT: seqjax.model.typing.Parameters,
    ObservationT: seqjax.model.typing.Observation,
](eqx.Module):
    latent_approximation: AmortizedVariationalApproximation[
        LatentStructT, tuple[ParameterStructT, ObservationT]
    ]
    parameter_approximation: UnconditionalVariationalApproximation[ParameterStructT]
    embedding: Embedder

    @abstractmethod
    def joint_sample_and_log_prob(
        self,
        observations: ObservationT,
        conditions: seqjax.model.typing.Condition,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
    ) -> tuple[
        ParameterStructT,
        jaxtyping.Float[jaxtyping.Array, "context_samples samples_per_context"],
        LatentStructT,
        jaxtyping.Float[
            jaxtyping.Array, "context_samples samples_per_context sample_length"
        ],
        jaxtyping.Int[jaxtyping.Array, "context_samples"],
        jaxtyping.Float[jaxtyping.Array, "context_samples sample_length"],
    ]: ...

    """
    Align parameters down the sample axis
    """

    @abstractmethod
    def buffer_params(self, parameters: ParameterStructT): ...


class FullAutoregressiveVI[
    LatentStructT: seqjax.model.typing.Particle,
    ParameterStructT: seqjax.model.typing.Parameters,
](SSMVariationalApproximation):
    def joint_sample_and_log_prob(
        self,
        observations,
        conditions,
        key,
        context_samples,
        samples_per_context,
    ):
        # all samples use the same context and weighting
        start_ix = jnp.zeros((context_samples), dtype=jnp.int32)
        latent_scaling = jnp.ones((context_samples, self.latent_approximation.shape[0]))

        parameter_keys, latent_keys = jnp.split(
            jrandom.split(key, (context_samples, samples_per_context * 2)), 2, axis=1
        )

        # vmap down both axes
        parameters, log_q_theta = jax.vmap(
            jax.vmap(self.parameter_approximation.sample_and_log_prob)
        )(parameter_keys)

        # vmap down the latent_keys and the parameter samples
        vmap_axes = [0, (0, None)]
        x_path, log_q_x_path = jax.vmap(
            jax.vmap(
                self.latent_approximation.sample_and_log_prob,
                in_axes=vmap_axes,
            ),
            in_axes=vmap_axes,
        )(
            latent_keys,
            (
                parameters.ravel(parameters),
                jnp.arange(self.latent_approximation.shape[0]).reshape(
                    -1, 1
                ),  # just give location information
            ),
        )

        return (parameters, log_q_theta, x_path, log_q_x_path, start_ix, latent_scaling)

    def buffer_params(self, parameters):
        sample_length = self.latent_approximation.shape[0]
        return buffer_params(
            parameters,
            jnp.zeros(sample_length),  # nothing in buffer
            parameters.batch_shape[0],
            parameters.batch_shape[1],
            sample_length,
        )


def sample_batch(key, sample_length, max_start_ix, y_path, condition):
    # each sample will come from the data replicated onto each device
    start_ix = jrandom.randint(key, (), 0, max_start_ix)
    samples = jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim, start_index=start_ix, slice_size=sample_length
        ),
        y_path,
    )
    csamples = jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim, start_index=start_ix, slice_size=sample_length
        ),
        condition,
    )

    return start_ix, samples, csamples


class BufferedSSMVI(eqx.Module):
    latent_approximation: AmortizedVariationalApproximation
    parameter_approximation: UnconditionalVariationalApproximation
    embedding: Embedder

    def joint_sample_and_log_prob(
        self,
        observations: seqjax.model.typing.Observation,
        conditions: seqjax.model.typing.Condition,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
    ) -> typing.Any:

        # read off configuration
        path_length = observations.batch_shape[0]
        sample_length = self.latent_approximation.shape[0]
        batch_length = self.latent_approximation.batch_length
        max_start_ix = path_length - batch_length - 1
        buffer_length = self.latent_approximation.buffer_length

        # split keys for sampling
        key, start_key = jrandom.split(key)
        parameter_keys, latent_keys = jnp.split(
            jrandom.split(key, (context_samples, samples_per_context * 2)), 2, axis=1
        )

        # vmap down both axes
        parameters, log_q_theta = jax.vmap(
            jax.vmap(self.parameter_approximation.sample_and_log_prob)
        )(parameter_keys)

        # sample observation slices
        # first padding out to give correct slices
        observations = jax.tree_util.tree_map(
            partial(
                jnp.pad,
                pad_width=(buffer_length, buffer_length),
                mode="mean",
            ),
            observations,
        )
        conditions = jax.tree_util.tree_map(
            partial(
                jnp.pad,
                pad_width=(buffer_length, buffer_length),
                mode="mean",
            ),
            conditions,
        )

        start_ix, observation_slices, condition_slices = jax.vmap(
            partial(
                sample_batch,
                sample_length=sample_length,
                max_start_ix=max_start_ix,
                y_path=observations,
                condition=conditions,
            )
        )(jrandom.split(start_key, context_samples))

        # vmap down the outer samples
        # then just latent_keys and the parameter samples
        x_path, log_q_x_path = jax.vmap(
            jax.vmap(
                self.latent_approximation.sample_and_log_prob, in_axes=[0, (0, None)]
            ),
        )(
            latent_keys,
            (
                jax.lax.stop_gradient(parameters.ravel(parameters)),
                jax.vmap(self.embedding.embed)(observation_slices),
            ),
        )
        # compute the latent scaling
        t_abs = start_ix.reshape(-1, 1) + jnp.arange(sample_length)
        num_possible_block = path_length - batch_length + 1
        block_membership_count = jnp.minimum(
            jnp.minimum(t_abs + 1, path_length - t_abs),
            min(batch_length, num_possible_block, path_length - batch_length + 1),
        )
        block_membership_count = jnp.maximum(block_membership_count, 1)
        latent_scaling = num_possible_block / block_membership_count
        sample_length = self.latent_approximation.shape[0]
        buffer_length = self.latent_approximation.buffer_length
        buffer_ix = jnp.arange(sample_length)
        batch_mask = jnp.logical_and(
            buffer_ix >= buffer_length,
            buffer_ix < sample_length - buffer_length,
        )

        latent_scaling = latent_scaling * batch_mask
        return (parameters, log_q_theta, x_path, log_q_x_path, start_ix, latent_scaling)

    def buffer_params(self, parameters):
        sample_length = self.latent_approximation.shape[0]
        buffer_length = self.latent_approximation.buffer_length
        buffer_ix = jnp.arange(sample_length)
        buffer_mask = jnp.logical_and(
            buffer_ix <= buffer_length,
            buffer_ix >= sample_length - buffer_length,
        )

        return buffer_params(
            parameters,
            buffer_mask,  # nothing in buffer
            parameters.batch_shape[0],
            parameters.batch_shape[1],
            sample_length,
        )
