import typing
from functools import partial

import equinox as eqx
import jaxtyping
import jax.numpy as jnp
import jax
import jax.random as jrandom

import seqjax.model.typing as seqjtyping
from seqjax.inference.interface import InferenceDataset
from seqjax.model.interface import BayesianSequentialModelProtocol
from seqjax.inference.vi.sampling import VISamplingKwargs
from seqjax.inference.particlefilter import SMCSampler
from seqjax.inference.score_estimator import buffered_score_estimate
from seqjax.inference.vi.interface import UnconditionalVariationalApproximation

class HybridSSMVI[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](eqx.Module):
    parameter_approximation: UnconditionalVariationalApproximation[InferenceParametersT]
    target_posterior: BayesianSequentialModelProtocol[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ]
    particle_filter: SMCSampler[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
    ]
    batch_length: int
    buffer_length: int

    def score_estimator(
        self, 
        dataset: InferenceDataset[ObservationT, ConditionT], 
        sample_kwargs: VISamplingKwargs,
        parameter: InferenceParametersT,
        key: jaxtyping.PRNGKeyArray,
    ):
        _estimate_score = jax.jit(
            partial(
                buffered_score_estimate,
                self.particle_filter,
                self.target_posterior,
                dataset,
                num_sequence_minibatch=sample_kwargs['num_sequence_minibatch'],
                buffer_length=self.buffer_length,
                batch_length=self.batch_length,
            )
        )
        return _estimate_score(parameter, key)

    def estimate_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        state: typing.Any = None,
        training: bool = False,
    ) -> tuple[typing.Any, typing.Any]:
        context_samples = sample_kwargs["context_samples"]
        samples_per_context = sample_kwargs["samples_per_context"]

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
        if context_samples != 1:
            raise ValueError(
                "HybridSSMVI does not support context_samples != 1. "
                f"Received context_samples={context_samples}."
            )

        score_key, param_key = jrandom.split(key)
        parameters, log_q_theta, next_state = jax.vmap(
            self.parameter_approximation.sample_and_log_prob,
            in_axes=(0, None, None),
            out_axes=(0, 0, None),
            axis_name="monte-carlo",
        )(
            jrandom.split(param_key, samples_per_context),
            None,
            state,
        )

        estimated_score = jax.vmap(
            self.score_estimator,
            in_axes=(None, None, 0, 0)
        )(
            dataset,
            sample_kwargs,
            parameters,
            jnp.array(jrandom.split(score_key, samples_per_context)),
        )

        score_term = jax.tree_util.tree_map(
            lambda e_score, param : jax.lax.stop_gradient(e_score) * (
                param - jax.lax.stop_gradient(param)
            ),
            estimated_score,
            parameters,
        )

        # for each sample, sum the leaf contributions
        score_leaves = jax.tree_util.tree_leaves(score_term)
        flat_score = sum(x.reshape(x.shape[0], -1).sum(axis=1) for x in score_leaves)
        pseudo_loss = log_q_theta - flat_score

        return jnp.mean(pseudo_loss), next_state

    def estimate_pretrain_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        state: typing.Any = None,
        training: bool = False,
    ) -> tuple[typing.Any, typing.Any]:
        raise Exception("No pretrain loss for Hybrid VI")

    def estimate_prior_fit_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        state: typing.Any = None,
    ) -> tuple[typing.Any, typing.Any]:
        context_samples = sample_kwargs["context_samples"]
        samples_per_context = sample_kwargs["samples_per_context"]
        parameter_keys = jrandom.split(key, (context_samples, samples_per_context))
        theta_q, log_q_theta, next_state = jax.vmap(
            jax.vmap(
                self.parameter_approximation.sample_and_log_prob,
                in_axes=(0, None, None),
                out_axes=(0, 0, None),
            ),
            in_axes=(0, None, None),
            out_axes=(0, 0, None),
        )(parameter_keys, None, state)
        log_p_theta = jax.vmap(
            jax.vmap(
                lambda x: self.target_posterior.parameterization.log_prob(x)
            )
        )(theta_q)
        prior_elbo = log_q_theta - log_p_theta
        return jnp.mean(prior_elbo), next_state
