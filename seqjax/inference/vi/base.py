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
from flowjax.bijections import Affine, AbstractBijection
from flowjax.bijections.masked_autoregressive import MaskedAutoregressive as FlowjaxMAF
from flowjax.distributions import (
    Normal as FlowjaxNormal,
    Transformed as FlowjaxTransformed,
)


import seqjax.model.typing as seqjtyping
from seqjax.inference.embedder import Embedder
from seqjax.model.evaluate import buffered_log_p_joint
from seqjax.model.base import BayesianSequentialModel


def _ensure_prng_key(key: jaxtyping.PRNGKeyArray) -> jaxtyping.PRNGKeyArray:
    """Convert legacy uint32 keys to typed JAX keys for FlowJax."""

    if (
        hasattr(key, "dtype")
        and getattr(key, "shape", ()) == (2,)
        and key.dtype == jnp.dtype("uint32")
    ):
        return jrandom.wrap_key_data(jnp.asarray(key, dtype=jnp.uint32))
    return key


class VariationalApproximation[
    TargetStructT: seqjtyping.Packable,
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
    TargetStructT: seqjtyping.Packable,
](VariationalApproximation[TargetStructT, tuple[jaxtyping.Array, jaxtyping.Array]]):
    batch_length: int
    buffer_length: int


class UnconditionalVariationalApproximation[
    TargetStructT: seqjtyping.Packable,
](VariationalApproximation[TargetStructT, None]):
    @abstractmethod
    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: None = None,
    ) -> tuple[TargetStructT, jaxtyping.Scalar]: ...


class VariationalApproximationFactory[
    TargetStructT: seqjtyping.Packable,
    ConditionT,
](typing.Protocol):
    def __call__(
        self, target_struct_cls: type[TargetStructT]
    ) -> VariationalApproximation[TargetStructT, ConditionT]: ...


class MeanField[TargetStructT: seqjtyping.Packable](
    UnconditionalVariationalApproximation[TargetStructT]
):
    target_struct_cls: type[TargetStructT]
    loc: jaxtyping.Float[jaxtyping.Array, " TargetStructT.dim"]
    _unc_scale: jaxtyping.Float[jaxtyping.Array, " TargetStructT.dim"]

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


class MaskedAutoregressiveFlow[
    TargetStructT: seqjtyping.Packable,
](UnconditionalVariationalApproximation[TargetStructT]):
    """Masked autoregressive flow over the flattened parameter space."""

    target_struct_cls: type[TargetStructT]
    base_distribution: FlowjaxNormal
    flow: FlowjaxMAF
    distribution: FlowjaxTransformed

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        super().__init__(target_struct_cls, shape=(target_struct_cls.flat_dim,))
        self.target_struct_cls = target_struct_cls
        dim = target_struct_cls.flat_dim

        loc = jnp.broadcast_to(jnp.asarray(base_loc), (dim,))
        scale = jnp.broadcast_to(jnp.asarray(base_scale), (dim,))
        self.base_distribution = FlowjaxNormal(loc, scale)

        if transformer is None:
            transformer = Affine()

        flow_key = _ensure_prng_key(key)
        self.flow = FlowjaxMAF(
            flow_key,
            transformer=transformer,
            dim=dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
        distribution = typing.cast(
            FlowjaxTransformed,
            FlowjaxTransformed(self.base_distribution, self.flow),  # type: ignore[arg-type, call-arg]
        )
        self.distribution = distribution

    def sample_and_log_prob(self, key, condition=None):
        flow_key = _ensure_prng_key(key)
        flat_sample = self.distribution.sample(flow_key)
        log_q = self.distribution.log_prob(flat_sample)
        return self.target_struct_cls.unravel(flat_sample), log_q


class MaskedAutoregressiveFlowFactory[TargetStructT: seqjtyping.Packable](
    VariationalApproximationFactory[TargetStructT, None]
):
    def __init__(
        self,
        *,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        self._key = key
        self._nn_width = nn_width
        self._nn_depth = nn_depth
        self._base_loc = base_loc
        self._base_scale = base_scale
        self._transformer = transformer

    def __call__(
        self, target_struct_cls: type[TargetStructT]
    ) -> MaskedAutoregressiveFlow[TargetStructT]:
        return MaskedAutoregressiveFlow(
            target_struct_cls,
            key=self._key,
            nn_width=self._nn_width,
            nn_depth=self._nn_depth,
            base_loc=self._base_loc,
            base_scale=self._base_scale,
            transformer=self._transformer,
        )


class AmortizedMaskedAutoregressiveFlow[
    TargetStructT: seqjtyping.Packable,
](AmortizedVariationalApproximation[TargetStructT]):
    """Conditional masked autoregressive flow over buffered latent paths."""

    target_struct_cls: type[TargetStructT]
    base_distribution: FlowjaxNormal
    flow: FlowjaxMAF
    distribution: FlowjaxTransformed
    conditioner: eqx.nn.MLP
    _flat_sample_dim: int = eqx.field(static=True)
    _condition_input_dim: int = eqx.field(static=True)
    _condition_dim: int = eqx.field(static=True)
    _parameter_dim: int = eqx.field(static=True)
    _context_dim: int = eqx.field(static=True)

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        buffer_length: int,
        batch_length: int,
        context_dim: int,
        parameter_dim: int,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        conditioner_width: int,
        conditioner_depth: int,
        conditioner_out_dim: int,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        sample_length = 2 * buffer_length + batch_length
        shape = (sample_length, target_struct_cls.flat_dim)
        super().__init__(
            target_struct_cls,
            shape=shape,
            batch_length=batch_length,
            buffer_length=buffer_length,
        )
        self.target_struct_cls = target_struct_cls
        flat_sample_dim = sample_length * target_struct_cls.flat_dim
        self._flat_sample_dim = flat_sample_dim
        self._parameter_dim = parameter_dim
        self._context_dim = context_dim

        if transformer is None:
            transformer = Affine()

        loc = jnp.broadcast_to(jnp.asarray(base_loc), (flat_sample_dim,))
        scale = jnp.broadcast_to(jnp.asarray(base_scale), (flat_sample_dim,))
        self.base_distribution = FlowjaxNormal(loc, scale)

        cond_input_dim = sample_length * (parameter_dim + context_dim)
        if cond_input_dim <= 0:
            raise ValueError("Conditioner input dimension must be positive")

        key = _ensure_prng_key(key)
        cond_key, flow_key = jrandom.split(key)
        self.conditioner = eqx.nn.MLP(
            in_size=cond_input_dim,
            out_size=conditioner_out_dim,
            width_size=conditioner_width,
            depth=conditioner_depth,
            key=cond_key,
        )
        self._condition_input_dim = cond_input_dim
        self._condition_dim = conditioner_out_dim

        self.flow = FlowjaxMAF(
            flow_key,
            transformer=transformer,
            dim=flat_sample_dim,
            cond_dim=conditioner_out_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
        distribution = typing.cast(
            FlowjaxTransformed,
            FlowjaxTransformed(self.base_distribution, self.flow),  # type: ignore[arg-type, call-arg]
        )
        self.distribution = distribution

    def _build_condition(
        self,
        theta_context: jaxtyping.Array,
        observation_context: jaxtyping.Array,
    ) -> jaxtyping.Array:
        sample_length = self.shape[0]
        theta_flat = jnp.ravel(theta_context)
        obs_flat = jnp.ravel(observation_context)

        expected_theta = sample_length * self._parameter_dim
        expected_obs = sample_length * self._context_dim

        if theta_flat.size < expected_theta or obs_flat.size < expected_obs:
            raise ValueError(
                "Condition tensors shorter than required for flow conditioning"
            )

        theta_features = theta_flat[:expected_theta]
        observation_features = obs_flat[:expected_obs]
        conditioning_input = jnp.concatenate(
            [theta_features, observation_features], axis=0
        )
        return self.conditioner(conditioning_input)

    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: tuple[jaxtyping.Array, jaxtyping.Array],
    ) -> tuple[TargetStructT, jaxtyping.Scalar]:
        theta_context, observation_context = condition
        cond = self._build_condition(theta_context, observation_context)
        flow_key = _ensure_prng_key(key)
        flat_sample = self.distribution.sample(flow_key, condition=cond)
        log_q = self.distribution.log_prob(flat_sample, condition=cond)
        reshaped_sample = jnp.reshape(flat_sample, self.shape)
        latent_sample = typing.cast(
            TargetStructT, self.target_struct_cls.unravel(reshaped_sample)
        )
        return latent_sample, log_q


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
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](typing.Protocol):
    latent_approximation: AmortizedVariationalApproximation[ParticleT]
    parameter_approximation: UnconditionalVariationalApproximation[ParametersT]
    embedding: Embedder

    def joint_sample_and_log_prob(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
    ) -> tuple[
        ParametersT,
        jaxtyping.Float[jaxtyping.Array, "context_samples samples_per_context"],
        ParticleT,
        jaxtyping.Float[
            jaxtyping.Array, "context_samples samples_per_context sample_length"
        ],
        typing.Any,
    ]: ...

    def estimate_loss(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
        target_posterior: BayesianSequentialModel[
            ParticleT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
        ],
        hyperparameters: HyperParametersT,
    ) -> typing.Any: ...


class FullAutoregressiveVI[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](eqx.Module):
    latent_approximation: AmortizedVariationalApproximation[LatentT]
    parameter_approximation: UnconditionalVariationalApproximation[ParametersT]
    embedding: Embedder

    def joint_sample_and_log_prob(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
    ) -> tuple[
        ParametersT,
        jaxtyping.Float[jaxtyping.Array, "context_samples samples_per_context"],
        LatentT,
        jaxtyping.Float[
            jaxtyping.Array, "context_samples samples_per_context sample_length"
        ],
        typing.Any,
    ]:
        parameter_keys, latent_keys = jnp.split(
            jrandom.split(key, (context_samples, samples_per_context * 2)), 2, axis=1
        )

        # vmap down both axes
        parameters, log_q_theta = jax.vmap(
            jax.vmap(self.parameter_approximation.sample_and_log_prob)
        )(parameter_keys, None)
        theta_array = parameters.ravel(parameters)
        theta_array = jnp.tile(
            theta_array[:, :, None, :], (1, 1, observations.batch_shape[0], 1)
        )  # repeat down sequence axis for each sample

        # vmap down the latent_keys and the parameter samples
        vmap_axes = [0, (0, None)]
        context = jnp.linspace(-3, 3, self.latent_approximation.shape[0]).reshape(
            -1, 1
        )  # give location information
        embedded_context = jnp.hstack([context, self.embedding.embed(observations)])

        x_path, log_q_x_path = jax.vmap(
            jax.vmap(
                self.latent_approximation.sample_and_log_prob,
                in_axes=vmap_axes,
            ),
            in_axes=vmap_axes,
        )(
            latent_keys,
            (theta_array, embedded_context),
        )

        return (parameters, log_q_theta, x_path, log_q_x_path, None)

    def buffer_params(self, parameters: ParametersT, mask):
        theta_grad = parameters
        theta_no_grad = jax.lax.stop_gradient(parameters)

        return jax.tree.map(
            lambda a, b: jnp.where(mask, a, b),
            theta_grad,
            theta_no_grad,
        )

    def estimate_loss(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
        target_posterior: BayesianSequentialModel[
            LatentT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
        ],
        hyperparameters: HyperParametersT,
    ) -> typing.Any:
        theta_q, log_q_theta, x_path, log_q_x_path, extra_info = (
            self.joint_sample_and_log_prob(
                observations, conditions, key, context_samples, samples_per_context
            )
        )

        log_p_theta = jax.vmap(
            jax.vmap(
                lambda x: target_posterior.parameter_prior.log_prob(x, hyperparameters)
            )
        )(theta_q)

        batched_log_p_joint = jax.vmap(
            partial(buffered_log_p_joint, target_posterior.target),
            in_axes=[0, None, None, 0],
        )
        batched_log_p_joint = jax.vmap(batched_log_p_joint, in_axes=[0, None, None, 0])

        buffered_theta = jax.vmap(
            jax.vmap(self.buffer_params, in_axes=[0, None]), in_axes=[0, None]
        )(theta_q, jnp.ones(observations.batch_shape[0]))

        log_p_y_x_path = batched_log_p_joint(
            x_path,
            observations,
            conditions,
            target_posterior.target_parameter(buffered_theta),
        )

        neg_elbo = (log_q_theta - log_p_theta) + jnp.sum(
            log_q_x_path - log_p_y_x_path, axis=-1
        )

        return jnp.mean(neg_elbo)


def sample_batch(key, sample_length, max_start_ix, y_path, condition):
    # each sample will come from the data replicated onto each device
    start_ix = jrandom.randint(key, (), 0, max_start_ix + 1)
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


def sample_batch_and_mask(
    key, sequence_length, batch_length, buffer_length, y_path, condition
):
    sample_length = batch_length + 2 * buffer_length
    pad_length = batch_length - 1

    # each sample will come from the data replicated onto each device
    padded_start_ix = jrandom.randint(key, (), 0, sequence_length + pad_length)

    # where the buffer would like to start, may fall outside of data
    buffer_start = padded_start_ix - pad_length - buffer_length
    # pshirt into possible starts for the latent approximation
    approx_start = jnp.clip(buffer_start, min=0, max=sequence_length - sample_length)

    # construct the mask for theta, this probably could be
    # written in a clearer way
    batch_index = padded_start_ix + jnp.arange(batch_length)
    padded_mask = jax.nn.one_hot(
        batch_index, num_classes=sequence_length + 2 * pad_length
    ).any(axis=0)
    in_data = jnp.concatenate(
        (
            jnp.full(pad_length, 0),
            jnp.full(sequence_length, 1),
            jnp.full(pad_length, 0),
        )
    )
    theta_mask = jnp.logical_and(in_data, padded_mask)[pad_length:-pad_length]
    theta_mask = jax.lax.dynamic_slice_in_dim(
        theta_mask, start_index=approx_start, slice_size=sample_length
    )

    samples = jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=approx_start,
            slice_size=sample_length,
        ),
        y_path,
    )
    csamples = jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=approx_start,
            slice_size=sample_length,
        ),
        condition,
    )

    return approx_start, samples, csamples, theta_mask


class BufferedSSMVI[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](eqx.Module):
    latent_approximation: AmortizedVariationalApproximation[ParticleT]
    parameter_approximation: UnconditionalVariationalApproximation[ParametersT]
    embedding: Embedder
    control_variate: bool = eqx.field(default=False, static=True)

    def joint_sample_and_log_prob(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
    ) -> tuple[
        ParametersT,
        jaxtyping.Float[jaxtyping.Array, "context_samples samples_per_context"],
        ParticleT,
        jaxtyping.Float[
            jaxtyping.Array, "context_samples samples_per_context sample_length"
        ],
        typing.Any,
    ]:
        # read off configuration
        path_length = observations.batch_shape[0]
        batch_length = self.latent_approximation.batch_length
        buffer_length = self.latent_approximation.buffer_length

        # split keys for sampling
        key, start_key = jrandom.split(key)
        parameter_keys, latent_keys = jnp.split(
            jrandom.split(key, (context_samples, samples_per_context * 2)), 2, axis=1
        )

        # vmap down both axes
        parameters, log_q_theta = jax.vmap(
            jax.vmap(self.parameter_approximation.sample_and_log_prob)
        )(parameter_keys, None)

        # sample batches and masks
        approx_start, y_batch, c_batch, theta_mask = jax.vmap(
            partial(
                sample_batch_and_mask,
                sequence_length=path_length,
                batch_length=batch_length,
                buffer_length=buffer_length,
                y_path=observations,
                condition=conditions,
            )
        )(jrandom.split(start_key, context_samples))

        buffered_theta_array = parameters.ravel(
            jax.vmap(jax.vmap(self.buffer_params, in_axes=[0, None]))(
                parameters, theta_mask
            )
        )
        buffered_theta_array = jax.lax.select(
            self.control_variate,
            buffered_theta_array,
            jax.lax.stop_gradient(buffered_theta_array),
        )

        # vmap down the outer samples
        # then just latent_keys and the parameter samples
        x_path, log_q_x_path = jax.vmap(
            jax.vmap(
                self.latent_approximation.sample_and_log_prob, in_axes=[0, (0, None)]
            ),
        )(
            latent_keys,
            (
                buffered_theta_array,
                jax.vmap(self.embedding.embed)(y_batch),
            ),
        )

        return (
            parameters,
            log_q_theta,
            x_path,
            log_q_x_path,
            (approx_start, theta_mask, y_batch, c_batch),
        )

    def buffer_params(self, parameters, mask):
        theta_grad = parameters
        theta_no_grad = jax.lax.stop_gradient(parameters)

        return jax.tree.map(
            lambda a, b: jnp.where(mask, a, b),
            theta_grad,
            theta_no_grad,
        )

    def estimate_loss(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
        target_posterior: BayesianSequentialModel[
            ParticleT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
        ],
        hyperparameters: HyperParametersT,
    ) -> typing.Any:
        # each index appears in max batch length batches
        # batches are sampled uniformly, so scale by
        latent_scaling = (
            self.latent_approximation.batch_length + observations.batch_shape[0]
        ) / self.latent_approximation.batch_length

        theta_q, log_q_theta, x_path, log_q_x_path, extra_info = (
            self.joint_sample_and_log_prob(
                observations, conditions, key, context_samples, samples_per_context
            )
        )
        _, theta_mask, y_batch, c_batch = extra_info

        log_p_theta = jax.vmap(
            jax.vmap(
                lambda x: target_posterior.parameter_prior.log_prob(x, hyperparameters)
            )
        )(theta_q)

        batched_log_p_joint = jax.vmap(
            partial(buffered_log_p_joint, target_posterior.target),
            in_axes=[0, None, None, 0],
        )
        batched_log_p_joint = jax.vmap(batched_log_p_joint, in_axes=[0, 0, 0, 0])

        buffered_theta = jax.vmap(jax.vmap(self.buffer_params, in_axes=[0, None]))(
            theta_q, theta_mask
        )

        log_p_y_x_path = batched_log_p_joint(
            x_path, y_batch, c_batch, target_posterior.target_parameter(buffered_theta)
        )

        if log_q_x_path.ndim == log_p_y_x_path.ndim:
            latent_terms = jnp.sum(log_q_x_path - log_p_y_x_path, axis=-1)
        else:
            latent_terms = log_q_x_path - jnp.sum(log_p_y_x_path, axis=-1)

        neg_elbo = (log_q_theta - log_p_theta) + latent_scaling * latent_terms

        return jnp.mean(neg_elbo)

    def estimate_pretrain_loss(
        self,
        observations: ObservationT,
        conditions: ConditionT,
        key: jaxtyping.PRNGKeyArray,
        context_samples: int,
        samples_per_context: int,
        target_posterior: BayesianSequentialModel,
        hyperparameters: HyperParametersT,
    ) -> typing.Any:
        # read off configuration
        path_length = observations.batch_shape[0]
        batch_length = self.latent_approximation.batch_length
        buffer_length = self.latent_approximation.buffer_length

        # split keys for sampling
        key, start_key = jrandom.split(key)
        parameter_keys, latent_keys = jnp.split(
            jrandom.split(key, (context_samples, samples_per_context * 2)), 2, axis=1
        )

        # vmap down both axes
        parameters = jax.vmap(jax.vmap(target_posterior.parameter_prior.sample))(
            parameter_keys, None
        )

        # sample batches and masks
        approx_start, y_batch, c_batch, theta_mask = jax.vmap(
            partial(
                sample_batch_and_mask,
                sequence_length=path_length,
                batch_length=batch_length,
                buffer_length=buffer_length,
                y_path=observations,
                condition=conditions,
            )
        )(jrandom.split(start_key, context_samples))

        buffered_theta = jax.vmap(jax.vmap(self.buffer_params, in_axes=[0, None]))(
            parameters, theta_mask
        )
        buffered_theta_array = parameters.ravel(buffered_theta)

        # vmap down the outer samples
        # then just latent_keys and the parameter samples
        x_path, log_q_x_path = jax.vmap(
            jax.vmap(
                self.latent_approximation.sample_and_log_prob, in_axes=[0, (0, None)]
            ),
        )(
            latent_keys,
            (
                buffered_theta_array,
                jax.vmap(self.embedding.embed)(y_batch),
            ),
        )

        batched_log_p_joint = jax.vmap(
            partial(buffered_log_p_joint, target_posterior.target),
            in_axes=[0, None, None, 0],
        )
        batched_log_p_joint = jax.vmap(batched_log_p_joint, in_axes=[0, 0, 0, 0])

        log_p_y_x_path = batched_log_p_joint(
            x_path, y_batch, c_batch, target_posterior.target_parameter(buffered_theta)
        )

        if log_q_x_path.ndim == log_p_y_x_path.ndim:
            neg_elbo = jnp.sum(log_q_x_path - log_p_y_x_path, axis=-1)
        else:
            neg_elbo = log_q_x_path - jnp.sum(log_p_y_x_path, axis=-1)

        return jnp.mean(neg_elbo)
