import time
import typing
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import jaxtyping
from jax_tqdm import scan_tqdm  # type: ignore[import-not-found]

import seqjax.model.typing as seqjtyping
from seqjax import util
from seqjax.inference.interface import InferenceDataset, inference_method
from seqjax.inference.particlefilter import run_filter
from seqjax.inference.particlefilter import registry as particle_filter_registry
from seqjax.inference.particlefilter.base import FilterData
from seqjax.inference.vi.base import _sample_sequence_minibatch, sample_batch_and_mask
from seqjax.model.base import BayesianSequentialModel


class SGLDConfig[
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
):
    """Configuration for :func:`run_sgld`."""

    particle_filter_config: particle_filter_registry.BootstrapFilterConfig
    step_size: float | ParametersT = 1e-3
    num_steps: int = 100
    initial_parameter_guesses: int = 20
    num_sequence_minibatch: int = 1
    time_limit_s: None | float = None
    sample_block_size: int = 1000


class BufferedSGLDConfig[
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
):
    particle_filter_config: particle_filter_registry.BootstrapFilterConfig
    step_size: float | ParametersT = 1e-3
    num_steps: None | int = 5000
    num_sequence_minibatch: int = 1
    time_limit_s: None | float = None
    buffer_length: int = 5
    batch_length: int = 10
    sample_block_size: int = 1000


def _tree_randn_like[ParametersT: seqjtyping.Parameters](
    key: jaxtyping.PRNGKeyArray, tree: ParametersT
) -> ParametersT:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jrandom.split(key, len(leaves))
    new_leaves = [
        jrandom.normal(k, shape=jnp.shape(leaf)) for k, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def estimate_initial_step_score(model, filter_data: FilterData):
    # grad wrt to parameters
    def model_initial_log_prob(particles, inference_parameters):
        model_parameters = model.convert_to_model_parameters(inference_parameters)
        return model.target.emission.log_prob(
            particles,
            filter_data.observation,
            (),
            filter_data.condition,
            model_parameters,
        ) + model.target.prior.log_prob(
            particles,
            filter_data.condition,
            model_parameters,
        )

    model_initial_score = jax.grad(model_initial_log_prob, argnums=-1)

    return jax.vmap(
        model_initial_score,
        in_axes=(0, None),
    )(filter_data.particles, filter_data.inference_parameters)


def estimate_step_score(model, filter_data: FilterData):
    # grad wrt to parameters
    def model_step_log_prob(
        particles, emission_particles, transition_history, inference_parameters
    ):
        model_parameters = model.convert_to_model_parameters(inference_parameters)
        return model.target.emission.log_prob(
            emission_particles,
            filter_data.observation,
            (),
            filter_data.condition,
            model_parameters,
        ) + model.target.transition.log_prob(
            transition_history,
            particles,
            filter_data.condition,
            model_parameters,
        )

    model_step_score = jax.grad(model_step_log_prob, argnums=-1)

    transition_history = model.target.latent_view_for_transition(filter_data.resampled_particles)
    emission_history = (
        filter_data.resampled_particles[-(model.target.emission.order - 1) :]
        if model.target.emission.order > 1
        else ()
    )
    emission_particles = (*emission_history, filter_data.particles[-1])
    proposed_particles = filter_data.particles[-1]

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
        model: BayesianSequentialModel[
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
    model: BayesianSequentialModel[
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
        convert_to_model_parameters=model.convert_to_model_parameters,
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


def run_sgld[ParametersT: seqjtyping.Parameters](
    grad_estimator: typing.Callable[[ParametersT, jaxtyping.PRNGKeyArray], ParametersT],
    key: jaxtyping.PRNGKeyArray,
    initial_parameters: ParametersT,
    config: SGLDConfig | BufferedSGLDConfig,
    num_samples: int,
    noise_rescale: float = 1.0,
) -> ParametersT:
    """Run SGLD updates using ``grad_estimator``."""

    split_keys = jrandom.split(key, 2 * num_samples)
    grad_keys = split_keys[:num_samples]
    noise_keys = split_keys[num_samples:]

    if jax.tree_util.tree_structure(config.step_size) == jax.tree_util.tree_structure(
        initial_parameters
    ):  # type: ignore[operator]
        step_sizes = config.step_size
    else:
        step_sizes = jax.tree_util.tree_map(lambda _: config.step_size, initial_parameters)

    @scan_tqdm(num_samples)
    def step(
        carry: tuple[int, ParametersT],
        inp: tuple[jaxtyping.PRNGKeyArray, jaxtyping.PRNGKeyArray],
    ):
        ix, params = carry
        _, (g_key, n_key) = inp
        grad = grad_estimator(params, g_key)
        noise = _tree_randn_like(n_key, params)
        updates = jax.tree_util.tree_map(
            lambda g, n, s: s * g + jnp.sqrt(2.0 * s) * n * noise_rescale,
            grad,
            noise,
            step_sizes,
        )
        params = eqx.apply_updates(params, updates)
        return (ix + 1, params), params

    step = typing.cast(
        typing.Callable[
            [
                tuple[int, ParametersT],
                tuple[
                    jaxtyping.Array,
                    tuple[jaxtyping.PRNGKeyArray, jaxtyping.PRNGKeyArray],
                ],
            ],
            tuple[
                tuple[int, ParametersT],
                ParametersT,
            ],
        ],
        step,
    )

    samples: ParametersT = jax.lax.scan(
        step,
        (0, initial_parameters),
        # scan_tqdm consumes the first tuple entry
        (jnp.arange(num_samples), (grad_keys, noise_keys)),
    )[1]
    return samples


@inference_method
def run_full_sgld_mcmc[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    hyperparameters: HyperParametersT,
    key: jaxtyping.PRNGKeyArray,
    dataset: InferenceDataset[ObservationT, ConditionT],
    test_samples: int,
    config: SGLDConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    if config.num_sequence_minibatch <= 0:
        raise ValueError(
            "SGLDConfig.num_sequence_minibatch must be positive. "
            f"Received {config.num_sequence_minibatch}."
        )
    if config.num_sequence_minibatch > dataset.num_sequences:
        raise ValueError(
            "SGLDConfig.num_sequence_minibatch cannot exceed dataset.num_sequences. "
            f"Received num_sequence_minibatch={config.num_sequence_minibatch}, "
            f"dataset.num_sequences={dataset.num_sequences}."
        )

    particle_filter = particle_filter_registry._build_filter(
        target_posterior, config.particle_filter_config
    )

    conditions = dataset.conditions
    sequence_score_estimator: SequenceScoreEstimator = _estimate_sequence_score

    def _estimate_score(particle_filter, model, params, grad_key):
        minibatch_key, sequence_pf_keys_key = jrandom.split(grad_key)
        sampled_observations, sampled_conditions, sequence_minibatch_rescaling = (
            _sample_sequence_minibatch(
                dataset,
                minibatch_key,
                config.num_sequence_minibatch,
            )
        )
        sequence_pf_keys = jrandom.split(
            sequence_pf_keys_key,
            config.num_sequence_minibatch,
        )

        if isinstance(conditions, seqjtyping.NoCondition):
            minibatch_likelihood_score = jax.vmap(
                lambda sequence_key, sequence_observation: sequence_score_estimator(
                    particle_filter,
                    model,
                    params,
                    sequence_key,
                    sequence_observation,
                    seqjtyping.NoCondition(),
                )
            )(sequence_pf_keys, sampled_observations)
        else:
            minibatch_likelihood_score = jax.vmap(
                lambda sequence_key, sequence_observation, sequence_condition: sequence_score_estimator(
                    particle_filter,
                    model,
                    params,
                    sequence_key,
                    sequence_observation,
                    sequence_condition,
                )
            )(sequence_pf_keys, sampled_observations, sampled_conditions)

        rescaled_likelihood_score = jax.tree_util.tree_map(
            lambda score_leaf: sequence_minibatch_rescaling * jnp.sum(score_leaf, axis=0),
            minibatch_likelihood_score,
        )

        log_prior_score = jax.grad(model.parameter_prior.log_prob, argnums=0)(
            params,
            hyperparameters,
        )

        return jax.tree_util.tree_map(
            lambda prior_leaf, likelihood_leaf: prior_leaf + likelihood_leaf,
            log_prior_score,
            rescaled_likelihood_score,
        )

    inference_time_start = time.time()
    init_key, next_sample_key = jrandom.split(key)
    initial_parameters = target_posterior.parameter_prior.sample(init_key, hyperparameters)

    # by default sample in chunks of 1000
    num_samples = config.sample_block_size
    sample_blocks = [
        jax.tree_util.tree_map(partial(jnp.expand_dims, axis=0), initial_parameters)
    ]
    samples_taken = 0
    block_times_s = []
    while True:
        sample_key, next_sample_key = jrandom.split(next_sample_key)
        start_parameter = util.index_pytree(sample_blocks[-1], -1)
        samples = run_sgld(
            jax.jit(partial(_estimate_score, model=target_posterior)),
            sample_key,
            start_parameter,
            config,
            num_samples,
        )
        samples_taken += num_samples

        elapsed_time_s = time.time() - inference_time_start
        block_times_s.append((elapsed_time_s, samples_taken))
        sample_blocks.append(samples)

        if config.time_limit_s and elapsed_time_s > config.time_limit_s:
            print("Stopping due to time limit")
            break

        if config.num_steps and samples_taken >= config.num_steps:
            print("Stopping due to sample limit")
            break

        print(f"Elapsed time: {int(elapsed_time_s / 60)} minutes")

    all_samples = util.concat_pytree(*sample_blocks)
    return all_samples, block_times_s


@inference_method
def run_buffer_sgld_mcmc[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    hyperparameters: HyperParametersT,
    key: jaxtyping.PRNGKeyArray,
    dataset: InferenceDataset[ObservationT, ConditionT],
    test_samples: int,
    config: BufferedSGLDConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    if config.num_sequence_minibatch <= 0:
        raise ValueError(
            "BufferedSGLDConfig.num_sequence_minibatch must be positive. "
            f"Received {config.num_sequence_minibatch}."
        )
    if config.num_sequence_minibatch > dataset.num_sequences:
        raise ValueError(
            "BufferedSGLDConfig.num_sequence_minibatch cannot exceed dataset.num_sequences. "
            f"Received num_sequence_minibatch={config.num_sequence_minibatch}, "
            f"dataset.num_sequences={dataset.num_sequences}."
        )

    particle_filter = particle_filter_registry._build_filter(
        target_posterior,
        config.particle_filter_config,
    )
    conditions = dataset.conditions

    def _buffered_estimate_score(
        particle_filter,
        model,
        dataset,
        params,
        grad_key,
    ):
        minibatch_key, start_keys_key, sequence_pf_keys_key = jrandom.split(grad_key, 3)
        sampled_observations, sampled_conditions, sequence_minibatch_rescaling = (
            _sample_sequence_minibatch(
                dataset,
                minibatch_key,
                config.num_sequence_minibatch,
            )
        )
        sequence_pf_keys = jrandom.split(
            sequence_pf_keys_key,
            config.num_sequence_minibatch,
        )
        start_keys = jrandom.split(start_keys_key, config.num_sequence_minibatch)

        path_length = sampled_observations.batch_shape[1]
        latent_scaling = (config.batch_length + path_length - 1) / config.batch_length

        approx_start, y_batch, c_batch, theta_mask = jax.vmap(
            sample_batch_and_mask,
            in_axes=(0, None, None, None, 0, 0),
        )(
            start_keys,
            path_length,
            config.batch_length,
            config.buffer_length,
            sampled_observations,
            sampled_conditions,
        )

        del approx_start

        if isinstance(conditions, seqjtyping.NoCondition):
            batched_out = jax.vmap(
                lambda sequence_key, sequence_obs: run_filter(
                    sequence_key,
                    particle_filter,
                    params,
                    sequence_obs,
                    condition_path=seqjtyping.NoCondition(),
                    recorders=(
                        partial(estimate_score_increment, model),
                        lambda fd: fd.ancestor_ix,
                    ),
                    convert_to_model_parameters=model.convert_to_model_parameters,
                )
            )(sequence_pf_keys, y_batch)
        else:
            batched_out = jax.vmap(
                lambda sequence_key, sequence_obs, sequence_cond: run_filter(
                    sequence_key,
                    particle_filter,
                    params,
                    sequence_obs,
                    condition_path=sequence_cond,
                    recorders=(
                        partial(estimate_score_increment, model),
                        lambda fd: fd.ancestor_ix,
                    ),
                    convert_to_model_parameters=model.convert_to_model_parameters,
                )
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
            lambda leaf: sequence_minibatch_rescaling * jnp.sum(leaf, axis=0),
            minibatch_likelihood_score,
        )

        log_prior_score = jax.grad(model.parameter_prior.log_prob, argnums=0)(
            params,
            hyperparameters,
        )

        return jax.tree_util.tree_map(
            lambda prior_leaf, likelihood_leaf: (prior_leaf + likelihood_leaf) / path_length,
            log_prior_score,
            rescaled_likelihood_score,
        )

    score_estimator = jax.jit(
        partial(
            _buffered_estimate_score,
            particle_filter,
            target_posterior,
            dataset,
        )
    )

    inference_time_start = time.time()
    init_key, next_sample_key = jrandom.split(key)
    initial_parameters = target_posterior.parameter_prior.sample(init_key, hyperparameters)

    num_samples = config.sample_block_size
    sample_blocks = [
        jax.tree_util.tree_map(partial(jnp.expand_dims, axis=0), initial_parameters)
    ]
    samples_taken = 0
    block_times_s = []
    while True:
        sample_key, next_sample_key = jrandom.split(next_sample_key)
        start_parameter = util.index_pytree(sample_blocks[-1], -1)
        samples = run_sgld(
            score_estimator,
            sample_key,
            start_parameter,
            config,
            num_samples,
            noise_rescale=1 / dataset.sequence_length,  # this is what Aicher does
        )
        samples_taken += num_samples

        elapsed_time_s = time.time() - inference_time_start
        block_times_s.append((elapsed_time_s, samples_taken))
        sample_blocks.append(samples)

        if config.time_limit_s and elapsed_time_s > config.time_limit_s:
            print("Stopping due to time limit")
            break

        if config.num_steps and samples_taken >= config.num_steps:
            print("Stopping due to sample limit")
            break

        print(f"Elapsed time: {int(elapsed_time_s / 60)} minutes")

    all_samples = util.concat_pytree(*sample_blocks)
    return all_samples, block_times_s
