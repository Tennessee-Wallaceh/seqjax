import typing
from functools import partial

import equinox as eqx
import jax.scipy.stats as jstats
import jax.scipy.linalg as jlinalg
import jaxtyping

import jax.numpy as jnp
import jax
from jax.nn import softplus
import jax.random as jrandom

import seqjax.model.typing as seqjtyping
from seqjax.inference.interface import InferenceDataset
from seqjax.model.evaluate import log_prob_joint
from seqjax.model.base import BayesianSequentialModel
from seqjax.inference.vi.sampling import VISamplingKwargs
from .api import LatentContext, Embedder, AmortizedVariationalApproximation, UnconditionalVariationalApproximation

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


class MultivariateNormal[TargetStructT: seqjtyping.Packable](
    UnconditionalVariationalApproximation[TargetStructT]
):
    """Full-covariance multivariate Gaussian over flattened parameters."""

    target_struct_cls: type[TargetStructT]
    loc: jaxtyping.Float[jaxtyping.Array, " TargetStructT.dim"]
    _unc_tril: jaxtyping.Float[jaxtyping.Array, "TargetStructT.dim TargetStructT.dim"]
    _diag_jitter: float = eqx.field(static=True)

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        diag_jitter: float = 1e-6,
    ):
        super().__init__(target_struct_cls, shape=(target_struct_cls.flat_dim,))
        self.target_struct_cls = target_struct_cls
        dim = target_struct_cls.flat_dim
        self.loc = jnp.zeros((dim,))
        self._unc_tril = jnp.zeros((dim, dim))
        self._diag_jitter = diag_jitter

    def _cholesky(self) -> jaxtyping.Array:
        strictly_lower = jnp.tril(self._unc_tril, k=-1)
        diagonal = softplus(jnp.diag(self._unc_tril)) + self._diag_jitter
        return strictly_lower + jnp.diag(diagonal)

    def sample_and_log_prob(self, key, condition=None):
        dim = self.target_struct_cls.flat_dim
        chol = self._cholesky()
        epsilon = jrandom.normal(key, (dim,))
        flat_sample = self.loc + chol @ epsilon

        centered = flat_sample - self.loc
        standardized = jlinalg.solve_triangular(chol, centered, lower=True)
        log_det = jnp.sum(jnp.log(jnp.diag(chol)))
        log_norm = -0.5 * dim * jnp.log(2.0 * jnp.pi)
        log_prob = log_norm - log_det - 0.5 * jnp.sum(standardized**2)
        return self.target_struct_cls.unravel(flat_sample), log_prob

    
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



def _sample_sequence_minibatch[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
](
    dataset: InferenceDataset[ObservationT, ConditionT],
    key: jaxtyping.PRNGKeyArray,
) -> tuple[ObservationT, ConditionT]:
    minibatch_index = jrandom.choice(key, dataset.num_sequences)

    sampled_observations = jax.tree_util.tree_map(
        lambda leaf: leaf[minibatch_index],
        dataset.observations,
    )
    sampled_conditions = jax.tree_util.tree_map(
        lambda leaf: leaf[minibatch_index],
        dataset.conditions,
    )

    return sampled_observations, sampled_conditions


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
    target_posterior: BayesianSequentialModel[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ]

    def joint_sample_and_log_prob(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
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
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        hyperparameters: HyperParametersT,
    ) -> typing.Any: ...


class FullVI[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](eqx.Module):
    latent_approximation: AmortizedVariationalApproximation[LatentT]
    parameter_approximation: UnconditionalVariationalApproximation[ParametersT]
    embedding: Embedder[ObservationT, ConditionT, ParametersT]

    def joint_sample_and_log_prob(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
    ) -> tuple[
        ParametersT,
        jaxtyping.Float[jaxtyping.Array, "context_samples samples_per_context"],
        LatentT,
        jaxtyping.Float[
            jaxtyping.Array, "context_samples samples_per_context sample_length"
        ],
        typing.Any,
    ]:
        context_samples = sample_kwargs["context_samples"]
        samples_per_context = sample_kwargs["samples_per_context"]
        num_sequence_minibatch = sample_kwargs["num_sequence_minibatch"]

        if context_samples != 1:
            raise ValueError(
                "FullVI does not support context_samples != 1. "
                f"Received context_samples={context_samples}."
            )

        key, sequence_key = jrandom.split(key)
        sampled_observations, sampled_conditions, sequence_minibatch_rescaling = _sample_sequence_minibatch(
            dataset,
            sequence_key,
            num_sequence_minibatch,
        )

        num_selected_sequences = sampled_observations.batch_shape[0]
        parameter_keys, latent_keys = jnp.split(
            jrandom.split(key, (num_selected_sequences, samples_per_context * 2)),
            2,
            axis=1,
        )

        parameters, log_q_theta = jax.vmap(
            jax.vmap(self.parameter_approximation.sample_and_log_prob)
        )(parameter_keys, None)
        latent_context = jax.vmap(self.embedding.embed)(
            sampled_observations,
            sampled_conditions,
            jax.lax.stop_gradient(parameters),
        )

        x_path, log_q_x = jax.vmap(
            jax.vmap(
                self.latent_approximation.sample_and_log_prob,
                in_axes=(
                    0,
                    typing.cast(
                        typing.Any,
                        LatentContext(
                            observation_context=None,
                            condition_context=None,
                            parameter_context=0,
                            embedded_context=typing.cast(jaxtyping.Array, None),
                            sequence_embedded_context=typing.cast(jaxtyping.Array, None),
                        ),
                    ),
                ),
            ),
            in_axes=(0, 0),
        )(
            latent_keys,
            latent_context,
        )

        return (
            parameters,
            log_q_theta,
            x_path,
            log_q_x,
            (sampled_observations, sampled_conditions, sequence_minibatch_rescaling),
        )

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
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
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
                dataset,
                key,
                sample_kwargs,
            )
        )
        sampled_observations, sampled_conditions, sequence_minibatch_rescaling = extra_info

        log_p_theta = jax.vmap(
            jax.vmap(
                lambda x: target_posterior.parameter_prior.log_prob(x, hyperparameters)
            )
        )(theta_q)

        batched_log_p_joint = jax.vmap(
            partial(log_prob_joint, target_posterior.target),
            in_axes=[0, 0, 0, 0],
        )
        batched_log_p_joint = jax.vmap(batched_log_p_joint, in_axes=[0, 0, 0, 0])

        sample_length = sampled_observations.batch_shape[1]
        buffered_theta = jax.vmap(
            jax.vmap(self.buffer_params, in_axes=[0, None]), in_axes=[0, None]
        )(theta_q, jnp.ones(sample_length))

        log_p_y_x = batched_log_p_joint(
            x_path,
            sampled_observations,
            sampled_conditions,
            target_posterior.convert_to_model_parameters(buffered_theta),
        )

        neg_elbo = (log_q_theta - log_p_theta) / sequence_minibatch_rescaling
        neg_elbo = neg_elbo + log_q_x_path - log_p_y_x

        return jnp.mean(neg_elbo)

    def estimate_pretrain_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        target_posterior: BayesianSequentialModel,
        hyperparameters: HyperParametersT,
    ) -> typing.Any:
        context_samples = sample_kwargs["context_samples"]
        samples_per_context = sample_kwargs["samples_per_context"]
        num_sequence_minibatch = sample_kwargs["num_sequence_minibatch"]

        key, sequence_key = jrandom.split(key)
        sampled_observations, sampled_conditions, _ = _sample_sequence_minibatch(
            dataset,
            sequence_key,
            num_sequence_minibatch,
        )

        if context_samples != 1:
            raise ValueError(
                "FullVI does not support context_samples != 1. "
                f"Received context_samples={context_samples}."
            )

        num_selected_sequences = sampled_observations.batch_shape[0]
        parameter_keys, latent_keys = jnp.split(
            jrandom.split(key, (num_selected_sequences, samples_per_context * 2)),
            2,
            axis=1,
        )

        parameters = jax.vmap(jax.vmap(target_posterior.parameter_prior.sample))(
            parameter_keys, None
        )

        latent_context = jax.vmap(self.embedding.embed)(
            sampled_observations,
            sampled_conditions,
            parameters,
        )

        x_path, log_q_x = jax.vmap(
            jax.vmap(
                self.latent_approximation.sample_and_log_prob,
                in_axes=(
                    0,
                    typing.cast(
                        typing.Any,
                        LatentContext(
                            observation_context=None,
                            condition_context=None,
                            parameter_context=0,
                            embedded_context=typing.cast(jaxtyping.Array, None),
                            sequence_embedded_context=typing.cast(jaxtyping.Array, None),
                        ),
                    ),
                ),
            ),
            in_axes=(0, 0),
        )(
            latent_keys,
            latent_context,
        )

        batched_log_p_joint = jax.vmap(
            partial(log_prob_joint, target_posterior.target),
            in_axes=[0, 0, 0, 0],
        )
        batched_log_p_joint = jax.vmap(batched_log_p_joint, in_axes=[0, 0, 0, 0])

        log_p_y_x = batched_log_p_joint(
            x_path,
            sampled_observations,
            sampled_conditions,
            target_posterior.convert_to_model_parameters(parameters),
        )

        neg_elbo = log_q_x - log_p_y_x
        return jnp.mean(neg_elbo)

    def estimate_prior_fit_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        target_posterior: BayesianSequentialModel,
        hyperparameters: HyperParametersT,
    ) -> typing.Any:
        context_samples = sample_kwargs["context_samples"]
        samples_per_context = sample_kwargs["samples_per_context"]

        parameter_keys = jrandom.split(key, (context_samples, samples_per_context))
        theta_q, log_q_theta = jax.vmap(
            jax.vmap(self.parameter_approximation.sample_and_log_prob)
        )(parameter_keys, None)
        log_p_theta = jax.vmap(
            jax.vmap(
                lambda x: target_posterior.parameter_prior.log_prob(x, hyperparameters)
            )
        )(theta_q)
        prior_elbo = log_q_theta - log_p_theta
        return jnp.mean(prior_elbo)


def sample_batch_and_mask(
    key, sequence_length, batch_length, buffer_length, observation_path, condition
):
    sample_length = batch_length + 2 * buffer_length
    pad_length = batch_length - 1

    # each sample will come from the data replicated onto each device
    padded_start_ix = jrandom.randint(key, (), 0, sequence_length + pad_length)

    # where the buffer would like to start, may fall outside of data
    buffer_start = padded_start_ix - pad_length - buffer_length
    # clip into possible starts for the latent approximation
    approx_start = jnp.clip(buffer_start, min=0, max=sequence_length - sample_length)

    # construct the mask for theta
    batch_start = padded_start_ix - pad_length  # may be negative (left padding)
    sample_index = approx_start + jnp.arange(sample_length)  # data indices covered by `samples`
    theta_mask = (sample_index >= batch_start) & (sample_index < batch_start + batch_length)
    theta_mask = theta_mask & (sample_index >= 0) & (sample_index < sequence_length)

    samples = jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=approx_start,
            slice_size=sample_length,
        ),
        observation_path,
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
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](eqx.Module):
    latent_approximation: AmortizedVariationalApproximation[LatentT]
    parameter_approximation: UnconditionalVariationalApproximation[InferenceParametersT]
    embedding: Embedder[ObservationT, ConditionT, InferenceParametersT]
    target_posterior: BayesianSequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ]
    batch_length: int
    buffer_length: int

    def sample_prior_and_latent_log_prob(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        seq_key: jaxtyping.PRNGKeyArray,
        subseq_key: jaxtyping.PRNGKeyArray,
        parameter_key: jaxtyping.PRNGKeyArray,
        latent_key: jaxtyping.PRNGKeyArray,
    ) -> tuple[
        InferenceParametersT,
        LatentT,
        jaxtyping.Float[jaxtyping.Scalar, ""],
        typing.Any,
    ]:
        observation_sequence, condition_sequence = _sample_sequence_minibatch(dataset, seq_key)
        approx_start, y_batch, c_batch, theta_mask = sample_batch_and_mask(
            subseq_key, 
            sequence_length=dataset.sequence_length,
            batch_length=self.batch_length,
            buffer_length=self.buffer_length,
            observation_path=observation_sequence, 
            condition=condition_sequence,
        )

        parameters = self.target_posterior.parameter_prior.sample(parameter_key, None)

        latent_context = self.embedding.embed(y_batch, c_batch, jax.lax.stop_gradient(parameters))

        x_path, log_q_x = self.latent_approximation.sample_and_log_prob(latent_key, latent_context)

        return (
            parameters,
            x_path,
            log_q_x,
            (approx_start, theta_mask, y_batch, c_batch),
        )
    
    def joint_sample_and_log_prob(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        seq_key: jaxtyping.PRNGKeyArray,
        subseq_key: jaxtyping.PRNGKeyArray,
        parameter_key: jaxtyping.PRNGKeyArray,
        latent_key: jaxtyping.PRNGKeyArray,
    ) -> tuple[
        InferenceParametersT,
        jaxtyping.Float[jaxtyping.Scalar, ""],
        LatentT,
        jaxtyping.Float[jaxtyping.Scalar, ""],
        typing.Any,
    ]:
        observation_sequence, condition_sequence = _sample_sequence_minibatch(dataset, seq_key)
        approx_start, y_batch, c_batch, theta_mask = sample_batch_and_mask(
            subseq_key, 
            sequence_length=dataset.sequence_length,
            batch_length=self.batch_length,
            buffer_length=self.buffer_length,
            observation_path=observation_sequence, 
            condition=condition_sequence,
        )

        parameters, log_q_theta = self.parameter_approximation.sample_and_log_prob(parameter_key, None)

        latent_context = self.embedding.embed(y_batch, c_batch, jax.lax.stop_gradient(parameters))

        x_path, log_q_x = self.latent_approximation.sample_and_log_prob(latent_key, latent_context)

        return (
            parameters,
            log_q_theta,
            x_path,
            log_q_x,
            (approx_start, theta_mask, y_batch, c_batch),
        )
    
    def batched_sample(
        self, 
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
    ):
        # vmaps builds the batch axes from innermost to outermost
        # so inner is latent+param samples for each (sequence, sub-sequence)
        # next is sub-sequence sample for each sequence sample
        # the outer most is sequence sample
        batched_sample = jax.vmap(self.joint_sample_and_log_prob, in_axes=(None, None, None, 0, 0))
        batched_sample = jax.vmap(batched_sample, in_axes=(None, None, 0, 0, 0))
        batched_sample = jax.vmap(batched_sample, in_axes=(None, 0, 0, 0, 0))
        seq_key, subseq_key, param_key, latent_key = jrandom.split(key, 4)
        return batched_sample(
            dataset,
            jrandom.split(seq_key, sample_kwargs['num_sequence_minibatch']),
            jrandom.split(
                subseq_key, 
                (
                    sample_kwargs['num_sequence_minibatch'], 
                    sample_kwargs['context_samples']
                ),
            ),
            jrandom.split(
                param_key, 
                (
                    sample_kwargs['num_sequence_minibatch'], 
                    sample_kwargs['context_samples'], 
                    sample_kwargs['samples_per_context']),
            ),
            jrandom.split(
                latent_key, 
                (
                    sample_kwargs['num_sequence_minibatch'], 
                    sample_kwargs['context_samples'], 
                    sample_kwargs['samples_per_context']),
            ),
        )

    def batched_pretrain_sample(
        self, 
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
    ):
        # vmaps builds the batch axes from innermost to outermost
        # so inner is latent+param samples for each (sequence, sub-sequence)
        # next is sub-sequence sample for each sequence sample
        # the outer most is sequence sample
        batched_sample = jax.vmap(self.sample_prior_and_latent_log_prob, in_axes=(None, None, None, 0, 0))
        batched_sample = jax.vmap(batched_sample, in_axes=(None, None, 0, 0, 0))
        batched_sample = jax.vmap(batched_sample, in_axes=(None, 0, 0, 0, 0))
        seq_key, subseq_key, param_key, latent_key = jrandom.split(key, 4)
        return batched_sample(
            dataset,
            jrandom.split(seq_key, sample_kwargs['num_sequence_minibatch']),
            jrandom.split(
                subseq_key, 
                (
                    sample_kwargs['num_sequence_minibatch'], 
                    sample_kwargs['context_samples']
                ),
            ),
            jrandom.split(
                param_key, 
                (
                    sample_kwargs['num_sequence_minibatch'], 
                    sample_kwargs['context_samples'], 
                    sample_kwargs['samples_per_context']),
            ),
            jrandom.split(
                latent_key, 
                (
                    sample_kwargs['num_sequence_minibatch'], 
                    sample_kwargs['context_samples'], 
                    sample_kwargs['samples_per_context']),
            ),
        )
    
    def batched_log_joint(self, *args, **kwargs):
        # input to the joint has leading batch axes 
        # [n_seq, n_subseq, n_mc]
        # in the inner we map down the MC param+latent samples that are for fixed
        # (sequence, sub-sequence)
        # then down the outer two axes
        batched_log_joint = jax.vmap(
            partial(log_prob_joint, self.target_posterior.target),
            in_axes=(0, None, None, 0) 
        )
        batched_log_joint = jax.vmap(batched_log_joint)
        batched_log_joint = jax.vmap(batched_log_joint)
        return batched_log_joint(*args, **kwargs)
    
    def batched_parameter_prior(self, *args, **kwargs):
        # parameter prior batches down each [n_seq, n_subseq, n_mc] batch axes
        batched_parameter_prior = jax.vmap(jax.vmap(jax.vmap(
            self.target_posterior.parameter_prior.log_prob
        )))
        return batched_parameter_prior(*args, **kwargs)
    
    def batched_buffer_params(self, parameters, mask):
        # each sample has been associated with a mask
        def _buffer_params(_parameters, _mask):
            return jax.tree.map(
                lambda a, b: jnp.where(_mask, a, b),
                _parameters,
                jax.lax.stop_gradient(_parameters),
            )
        batched_buffer = jax.vmap(jax.vmap(jax.vmap(_buffer_params)))
        return batched_buffer(parameters, mask)

    def estimate_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        hyperparameters: HyperParametersT,
    ) -> typing.Any:
        
        if sample_kwargs['num_sequence_minibatch'] <= 0:
            raise ValueError(
                "num_sequence_minibatch must be positive. "
                f"Received {sample_kwargs['num_sequence_minibatch']}."
            )
        if sample_kwargs['num_sequence_minibatch'] > dataset.num_sequences:
            raise ValueError(
                "num_sequence_minibatch cannot exceed dataset.num_sequences. "
                f"Received num_sequence_minibatch={sample_kwargs['num_sequence_minibatch']}, "
                f"dataset.num_sequences={dataset.num_sequences}."
            )
                    
        theta_q, log_q_theta, x_path, log_q_x, extra_info = self.batched_sample(
            dataset,
            key,
            sample_kwargs,

        )
        _, theta_mask, y_batch, c_batch = extra_info

        latent_scaling = (
            self.batch_length + dataset.sequence_length - 1
        ) / self.batch_length
        sequence_minibatch_rescaling = dataset.num_sequences / sample_kwargs['num_sequence_minibatch']

        log_p_theta = self.batched_parameter_prior(theta_q, hyperparameters)

        buffered_params = self.batched_buffer_params(theta_q, theta_mask)
        
        # print("log_p_theta", log_p_theta.shape)
        # print("log_q_theta", log_q_theta.shape)
        # print("x_path", x_path.batch_shape)
        # print("y_batch", y_batch.batch_shape)
        # print("theta_q", theta_q.batch_shape)
        # print("buffered_params", buffered_params.batch_shape)

        log_p_y_x = self.batched_log_joint(
            x_path,
            y_batch,
            c_batch,
            self.target_posterior.convert_to_model_parameters(buffered_params),
        )
        
        neg_elbo = (
            (log_q_x - log_p_y_x) 
            + (
                (log_q_theta - log_p_theta) 
                / (latent_scaling * sequence_minibatch_rescaling)
            )
        )
    
        return jnp.mean(neg_elbo)

    def estimate_pretrain_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
    ) -> typing.Any:
        theta_q, _, x_path, log_q_x, extra_info = self.batched_sample(
            dataset,
            key,
            sample_kwargs,

        )
        _, _, y_batch, c_batch = extra_info

        log_p_y_x = self.batched_log_joint(
            x_path,
            y_batch,
            c_batch,
            self.target_posterior.convert_to_model_parameters(theta_q),
        )

        neg_elbo = log_q_x - log_p_y_x

        return jnp.mean(neg_elbo)

    def estimate_prior_fit_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        hyperparameters: HyperParametersT,
    ) -> typing.Any:
        context_samples = sample_kwargs["context_samples"]
        samples_per_context = sample_kwargs["samples_per_context"]

        parameter_keys = jrandom.split(key, (context_samples, samples_per_context))
        theta_q, log_q_theta = jax.vmap(
            jax.vmap(self.parameter_approximation.sample_and_log_prob)
        )(parameter_keys, None)
        log_p_theta = jax.vmap(
            jax.vmap(
                lambda x: self.target_posterior.parameter_prior.log_prob(x, hyperparameters)
            )
        )(theta_q)
        prior_elbo = log_q_theta - log_p_theta
        return jnp.mean(prior_elbo)
