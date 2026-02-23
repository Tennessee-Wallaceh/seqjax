import time
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp

import equinox as eqx
import jaxtyping
from jax_tqdm import scan_tqdm  # type: ignore[import-not-found]

from seqjax import util
from seqjax.model.base import (
    BayesianSequentialModel,
)
import seqjax.model.typing as seqjtyping
from seqjax.inference.particlefilter import run_filter
from seqjax.inference.interface import inference_method
from functools import partial
from seqjax.inference.vi.base import sample_batch_and_mask
from seqjax.inference.particlefilter.base import FilterData
from seqjax.inference.particlefilter import registry as particle_filter_registry

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
            model_parameters
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
    def model_step_log_prob(particles, emission_particles, transition_history, inference_parameters):    
        model_parameters = model.convert_to_model_parameters(inference_parameters)
        return (
            model.target.emission.log_prob(
                emission_particles, 
                filter_data.observation, 
                (), 
                filter_data.condition, 
                model_parameters
            )
            + model.target.transition.log_prob(
                transition_history,
                particles,
                filter_data.condition,
                model_parameters,
            )
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
    )(proposed_particles, emission_particles, transition_history, filter_data.inference_parameters)

def estimate_score_increment(model, filter_data: FilterData):
    # accumulate increment associated with appropriate ancestor index, 
    # we must respect the histories
    return jax.lax.cond(
        filter_data.ancestor_ix[0] == -1,
        lambda fd: estimate_initial_step_score(model, fd),
        lambda fd: estimate_step_score(model, fd),
        filter_data,
    )


def run_sgld[ParametersT: seqjtyping.Parameters](
    grad_estimator: typing.Callable[[ParametersT, jaxtyping.PRNGKeyArray], ParametersT],
    key: jaxtyping.PRNGKeyArray,
    initial_parameters: ParametersT,
    config: SGLDConfig | BufferedSGLDConfig,
    num_samples: int,
    noise_rescale: float = 1.
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
        step_sizes = jax.tree_util.tree_map(
            lambda _: config.step_size, initial_parameters
        )

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
        (jnp.arange(num_samples), (grad_keys, noise_keys))
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
    observation_path: ObservationT,
    condition_path: ConditionT,
    test_samples: int,
    config: SGLDConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    particle_filter = particle_filter_registry._build_filter(
        target_posterior, config.particle_filter_config
    )

    def _estimate_score(particle_filter, model, params, key):
        out = run_filter(
            key,
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
            score_increment, ancestor_ix = inputs
            new_score = current_score[ancestor_ix] + score_increment
            return new_score, new_score

        def masked_score_for_leaf(ancestor_ix, norm_weights, leaf_score_increments):
            # leaf_score_increments [sample_length, num_particles]
            # mask [sample_length]
            final_score = jax.lax.scan(
                accumulate_scores,
                leaf_score_increments[0],
                (leaf_score_increments[1:], ancestor_ix[1:]),
            )[0]
            return jnp.sum(final_score * norm_weights)

        final_score = jax.tree_util.tree_map(
            partial(masked_score_for_leaf, ancestor_ix, norm_weights),
            score_increments
        )
        log_prior_score = jax.grad(model.parameter_prior.log_prob, argnums=0)(params, hyperparameters)

        return jax.tree_util.tree_map(
            lambda *x: sum(x),  log_prior_score, final_score
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


"""
Sketch implmentation for buffer sgld - needs checking
"""

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
    observation_path: ObservationT,
    condition_path: ConditionT,
    test_samples: int,
    config: BufferedSGLDConfig,
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    sequence_length = observation_path.batch_shape[0]

    particle_filter = particle_filter_registry._build_filter(
        target_posterior, config.particle_filter_config
    )
    def _buffered_estimate_score(
        particle_filter, 
        observation_path, 
        model, 
        batch_length, 
        buffer_length, 
        params, 
        key
    ):
        latent_scaling = (batch_length + observation_path.batch_shape[0] - 1) / batch_length

        batch_key, pf_key = jrandom.split(key)
        approx_start, y_batch, c_batch, theta_mask = sample_batch_and_mask(
            batch_key, 
            sequence_length,
            batch_length=batch_length,
            buffer_length=buffer_length,
            observation_path=observation_path,
            condition=seqjtyping.NoCondition()
        )

        out = run_filter(
            pf_key,
            particle_filter,
            params,
            y_batch,
            condition_path=c_batch,
            recorders=(
                partial(estimate_score_increment, model), 
                lambda fd: fd.ancestor_ix,
            ),
            convert_to_model_parameters=model.convert_to_model_parameters,
        )

        log_weights, _, (score_increments, ancestor_ix) = out
        norm_weights = jnp.exp(log_weights - jsp.special.logsumexp(log_weights))

        def accumulate_scores(current_score, inputs):
            score_increment, ancestor_ix = inputs
            new_score = current_score[ancestor_ix] + score_increment
            return new_score, new_score

        def masked_score_for_leaf(ancestor_ix, norm_weights, leaf_score_increments, mask):
            # leaf_score_increments [sample_length, num_particles]
            # mask [sample_length]
            score_increments = latent_scaling * jnp.expand_dims(mask, -1) * leaf_score_increments
            final_score = jax.lax.scan(
                accumulate_scores,
                score_increments[0],
                (score_increments[1:], ancestor_ix[1:]),
            )[0]
            return jnp.sum(final_score * norm_weights)

        final_score = jax.tree_util.tree_map(
            partial(masked_score_for_leaf, ancestor_ix, norm_weights, mask=theta_mask),
            score_increments
        )
        log_prior_score = jax.grad(model.parameter_prior.log_prob, argnums=0)(params, hyperparameters)
        # use rescaling in Aicher code
        return jax.tree_util.tree_map(
            lambda *x: sum(x) / sequence_length,  log_prior_score, final_score
        )

    score_esimator = jax.jit(partial(
        _buffered_estimate_score,
        particle_filter, 
        observation_path, 
        target_posterior,
        config.batch_length,
        config.buffer_length,
    ))

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
            score_esimator,
            sample_key,
            start_parameter,
            config,
            num_samples,
            noise_rescale=1 / sequence_length # this is what Aicher does 
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
