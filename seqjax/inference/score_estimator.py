import typing
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import jaxtyping

import seqjax.model.typing as seqjtyping
from seqjax.inference.particlefilter import run_filter
from seqjax.inference.particlefilter.interface import FilterData
from seqjax.inference.vi.base import _sample_sequence_minibatch, sample_batch_and_mask
from seqjax.model.interface import BayesianSequentialModelProtocol
from seqjax.model import util as model_util

def estimate_initial_step_score[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    model: BayesianSequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ], 
    filter_data: FilterData
):
    # grad wrt to parameters
    def model_initial_log_prob(particles, inference_parameters):
        model_parameters = model.parameterization.to_model_parameters(inference_parameters)
        return model.target.emission_log_prob(
            particles,
            filter_data.observation,
            model.target.observation_context(()),
            filter_data.condition,
            model_parameters,
        ) + model.target.prior_log_prob(
            particles,
            filter_data.condition,
            model_parameters,
        )

    model_initial_score = jax.grad(model_initial_log_prob, argnums=-1)

    return jax.vmap(
        model_initial_score,
        in_axes=(0, None),
    )(filter_data.particles, filter_data.inference_parameters)


def estimate_step_score[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    model: BayesianSequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ], 
    filter_data: FilterData[
        ParticleT,
        ObservationT,
        ConditionT,
        InferenceParametersT
    ]
):
    # grad wrt to parameters
    def model_step_log_prob(
        particles, emission_particles, transition_history, inference_parameters
    ):
        model_parameters = model.parameterization.to_model_parameters(inference_parameters)
        return model.target.emission_log_prob(
            emission_particles,
            filter_data.observation,
            model.target.observation_context(()),
            filter_data.condition,
            model_parameters,
        ) + model.target.transition_log_prob(
            transition_history,
            particles,
            filter_data.condition,
            model_parameters,
        )

    model_step_score = jax.grad(model_step_log_prob, argnums=-1)

    transition_history = model.target.latent_context(filter_data.resampled_particles.values)
    proposed_particles = filter_data.particles[-1]

    emission_particles = model_util.add_history(filter_data.resampled_particles, filter_data.particles[-1])

    return jax.vmap(
        model_step_score,
        in_axes=(0, 0, 0, None),
    )(
        proposed_particles,
        emission_particles,
        transition_history,
        filter_data.inference_parameters,
    )


def estimate_score_increment(model, filter_data: FilterData):
    # accumulate increment associated with appropriate ancestor index,
    # we must respect the histories
    return jax.lax.cond(
        filter_data.ancestor_ix[0] == -1,
        lambda fd: estimate_initial_step_score(model, fd),
        lambda fd: estimate_step_score(model, fd),
        filter_data,
    )


class SequenceScoreEstimator[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](typing.Protocol):
    def __call__(
        self,
        particle_filter,
        model: BayesianSequentialModelProtocol[
            ParticleT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
        ],
        params: InferenceParametersT,
        sequence_key: jaxtyping.PRNGKeyArray,
        observation_path: ObservationT,
        condition_path: ConditionT | seqjtyping.NoCondition,
    ) -> InferenceParametersT: ...


def _estimate_sequence_score[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    particle_filter,
    model: BayesianSequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    params: InferenceParametersT,
    sequence_key: jaxtyping.PRNGKeyArray,
    observation_path: ObservationT,
    condition_path: ConditionT | seqjtyping.NoCondition,
) -> InferenceParametersT:
    out = run_filter(
        sequence_key,
        particle_filter,
        params,
        observation_path,
        condition_path=condition_path,
        recorders=(
            partial(estimate_score_increment, model),
            lambda fd: fd.ancestor_ix,
        ),
    )

    log_weights, _, (score_increments, ancestor_ix) = out
    norm_weights = jnp.exp(log_weights - jsp.special.logsumexp(log_weights))

    def accumulate_scores(current_score, inputs):
        score_increment, current_ancestor_ix = inputs
        new_score = current_score[current_ancestor_ix] + score_increment
        return new_score, new_score

    def leaf_score(sequence_ancestor_ix, sequence_norm_weights, leaf_score_increments):
        final_score = jax.lax.scan(
            accumulate_scores,
            leaf_score_increments[0],
            (leaf_score_increments[1:], sequence_ancestor_ix[1:]),
        )[0]
        return jnp.sum(final_score * sequence_norm_weights)

    return jax.tree_util.tree_map(
        partial(leaf_score, ancestor_ix, norm_weights),
        score_increments,
    )

def buffered_score_estimate(
    particle_filter,
    model,
    dataset,
    params,
    grad_key,
    num_sequence_minibatch: int = 1,
    batch_length: int = 5,
    buffer_length: int = 5,
):
    minibatch_key, start_keys_key, sequence_pf_keys_key = jrandom.split(grad_key, 3)
    sampled_observations, sampled_conditions = (
        _sample_sequence_minibatch(
            dataset,
            minibatch_key,
            num_sequence_minibatch,
        )
    )
    sequence_pf_keys = jrandom.split(
        sequence_pf_keys_key,
        num_sequence_minibatch,
    )
    start_keys = jrandom.split(start_keys_key, num_sequence_minibatch)

    path_length = sampled_observations.batch_shape[1]
    latent_scaling = (batch_length + path_length - 1) / batch_length

    _, y_batch, c_batch, theta_mask = jax.vmap(
        sample_batch_and_mask,
        in_axes=(0, None, None, None, 0, 0),
    )(
        start_keys,
        path_length,
        batch_length,
        buffer_length,
        sampled_observations,
        sampled_conditions,
    )

    if isinstance(c_batch, seqjtyping.NoCondition):
        in_axes = (0, 0, None)
    else:
        in_axes = (0, 0, 0)

    batched_out = jax.vmap(
        lambda sequence_key, sequence_obs, c_batch: run_filter(
            sequence_key,
            particle_filter,
            params,
            sequence_obs,
            condition_path=c_batch,
            recorders=(
                partial(estimate_score_increment, model),
                lambda fd: fd.ancestor_ix,
            ),
        ),
        in_axes
    )(sequence_pf_keys, y_batch, c_batch)

    log_weights, _, (score_increments, ancestor_ix) = batched_out
    norm_weights = jnp.exp(
        log_weights - jsp.special.logsumexp(log_weights, axis=-1, keepdims=True)
    )

    def accumulate_scores(current_score, inputs):
        score_increment, current_ancestor_ix = inputs
        new_score = current_score[current_ancestor_ix] + score_increment
        return new_score, new_score

    def masked_score_for_leaf(
        minibatch_ancestor_ix,
        minibatch_norm_weights,
        minibatch_leaf_score_increments,
        minibatch_theta_mask,
    ):
        def per_sequence_masked_score(
            sequence_ancestor_ix,
            sequence_norm_weights,
            sequence_leaf_score_increments,
            sequence_mask,
        ):
            score_increment = (
                latent_scaling
                * jnp.expand_dims(sequence_mask, -1)
                * sequence_leaf_score_increments
            )
            final_score = jax.lax.scan(
                accumulate_scores,
                score_increment[0],
                (score_increment[1:], sequence_ancestor_ix[1:]),
            )[0]
            return jnp.sum(final_score * sequence_norm_weights)

        return jax.vmap(per_sequence_masked_score)(
            minibatch_ancestor_ix,
            minibatch_norm_weights,
            minibatch_leaf_score_increments,
            minibatch_theta_mask,
        )

    minibatch_likelihood_score = jax.tree_util.tree_map(
        partial(masked_score_for_leaf, ancestor_ix, norm_weights, minibatch_theta_mask=theta_mask),
        score_increments,
    )
    rescaled_likelihood_score = jax.tree_util.tree_map(
        lambda leaf: jnp.sum(leaf, axis=0),
        minibatch_likelihood_score,
    )

    log_prior_score = jax.grad(model.parameterization.log_prob, argnums=0)(
        params,
    )

    return jax.tree_util.tree_map(
        lambda prior_leaf, likelihood_leaf: (prior_leaf + likelihood_leaf),
        log_prior_score,
        rescaled_likelihood_score,
    )
