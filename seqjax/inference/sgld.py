import time
import typing
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jaxtyping
from jax_tqdm import scan_tqdm  # type: ignore[import-not-found]

import seqjax.model.typing as seqjtyping
from seqjax import util
from seqjax.inference.interface import InferenceDataset, inference_method
from seqjax.inference.particlefilter import registry as particle_filter_registry
from seqjax.inference.vi.base import _sample_sequence_minibatch, sample_batch_and_mask
from seqjax.model.interface import BayesianSequentialModelProtocol
from .score_estimator import buffered_score_estimate

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
    init_params: typing.Any = None


def _tree_randn_like[ParametersT: seqjtyping.Parameters](
    key: jaxtyping.PRNGKeyArray, tree: ParametersT
) -> ParametersT:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jrandom.split(key, len(leaves))
    new_leaves = [
        jrandom.normal(k, shape=jnp.shape(leaf)) for k, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)




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

    
    def grad_update(g, n, s):
        # g_max = 10000.
        # norm = jnp.linalg.norm(g)
        # scale = jnp.minimum(1.0, g_max / (norm + 1e-8))
        g = g
        return s * g * noise_rescale + jnp.sqrt(2.0 * s * noise_rescale) * n
    
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
            grad_update,
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
    target_posterior: BayesianSequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
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

    particle_filter = particle_filter_registry.build_filter(
        target_posterior, config.particle_filter_config
    )

    conditions = dataset.conditions
    sequence_score_estimator: SequenceScoreEstimator = _estimate_sequence_score

    def _estimate_score(particle_filter, model, params, grad_key):
        minibatch_key, sequence_pf_keys_key = jrandom.split(grad_key)
        sampled_observations, sampled_conditions = (
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
            lambda score_leaf:  jnp.sum(score_leaf, axis=0),
            minibatch_likelihood_score,
        )

        log_prior_score = jax.grad(model.parameterization.log_prob, argnums=0)(
            params,
        )

        return jax.tree_util.tree_map(
            lambda prior_leaf, likelihood_leaf: prior_leaf + likelihood_leaf,
            log_prior_score,
            rescaled_likelihood_score,
        )

    inference_time_start = time.time()
    init_key, next_sample_key = jrandom.split(key)
    initial_parameters = target_posterior.parameterization.sample(init_key)

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
    target_posterior: BayesianSequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
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

    particle_filter = particle_filter_registry.build_filter(
        target_posterior,
        config.particle_filter_config,
    )

    score_estimator = jax.jit(
        partial(
            buffered_score_estimate,
            particle_filter,
            target_posterior,
            dataset,
            num_sequence_minibatch=config.num_sequence_minibatch,
            buffer_length=config.buffer_length,
            batch_length=config.batch_length,
        )
    )

    inference_time_start = time.time()
    init_key, next_sample_key = jrandom.split(key)

    if config.init_params is None:
        initial_parameters = target_posterior.parameterization.sample(init_key)
    else:
        initial_parameters = config.init_params
        
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
