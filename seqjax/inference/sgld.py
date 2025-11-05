import time
import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import jaxtyping
from jax_tqdm import scan_tqdm  # type: ignore[import-not-found]

from seqjax import util
from seqjax.model.base import (
    BayesianSequentialModel,
)
import seqjax.model.typing as seqjtyping
from seqjax.inference.particlefilter import SMCSampler, run_filter, log_marginal
from seqjax.inference import particlefilter
from seqjax.inference.interface import inference_method


class SGLDConfig[
    LatentT: seqjtyping.Latent,
    InitialLatentT: tuple[seqjtyping.Latent, ...],
    ObservationLatentHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    eqx.Module,
):
    """Configuration for :func:`run_sgld`."""

    particle_filter: SMCSampler
    step_size: float | ParametersT = 1e-3
    num_samples: int = 100
    initial_parameter_guesses: int = 20


class BufferedSGLDConfig[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    eqx.Module,
):
    """Configuration for :func:`run_sgld`."""

    particle_filter: SMCSampler
    step_size: float | ParametersT = 1e-3
    num_samples: int = 100
    initial_parameter_guesses: int = 20
    buffer_length: int = 5
    batch_length: int = 10


def _tree_randn_like[ParametersT: seqjtyping.Parameters](
    key: jaxtyping.PRNGKeyArray, tree: ParametersT
) -> ParametersT:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jrandom.split(key, len(leaves))
    new_leaves = [
        jrandom.normal(k, shape=jnp.shape(leaf)) for k, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def build_score_increment(target_posterior: BayesianSequentialModel):
    def latent_prior_log_prob(particle, condition, parameters):
        return target_posterior.target.prior.log_prob(
            particle, condition, target_posterior.target_parameter(parameters)
        )

    latent_prior_score = jax.vmap(
        jax.grad(latent_prior_log_prob, argnums=2), in_axes=[0, None, None]
    )

    def observation_log_prob(
        particle, observation_history, observation, condition, parameters
    ):
        return target_posterior.target.emission.log_prob(
            particle,
            observation_history,
            observation,
            condition,
            target_posterior.target_parameter(parameters),
        )

    observation_score = jax.vmap(
        jax.grad(observation_log_prob, argnums=-1), in_axes=[0, None, None, None, None]
    )

    def transition_log_prob(particle_history, particle, condition, parameters):
        return target_posterior.target.transition.log_prob(
            particle_history,
            particle,
            condition,
            target_posterior.target_parameter(parameters),
        )

    transition_score = jax.vmap(
        jax.grad(transition_log_prob, argnums=-1), in_axes=[0, 0, None, None]
    )

    def score_increment(filter_data: particlefilter.base.FilterData):
        no_ancestor = jnp.all(filter_data.ancestor_ix == -1)

        ancestor_particles = jax.tree_util.tree_map(
            lambda leaf: jax.vmap(jax.lax.dynamic_index_in_dim, in_axes=[None, 0])(
                leaf, filter_data.ancestor_ix
            ).squeeze(),
            filter_data.last_particles,
        )

        def _latent_score(particles):
            return latent_prior_score(
                particles,
                filter_data.condition,
                filter_data.parameters,
            )

        def _transition_score(particles):
            return transition_score(
                ancestor_particles,
                particles[-1],
                filter_data.condition,
                filter_data.parameters,
            )

        scores = jax.tree_util.tree_map(
            lambda *xs: sum(xs),
            jax.lax.cond(
                no_ancestor, _latent_score, _transition_score, filter_data.particles
            ),
            observation_score(
                filter_data.particles,
                filter_data.obs_hist,
                filter_data.observation,
                filter_data.condition,
                filter_data.parameters,
            ),
        )

        return scores

    return score_increment


def run_sgld[ParametersT: seqjtyping.Parameters](
    grad_estimator: typing.Callable[[ParametersT, jaxtyping.PRNGKeyArray], ParametersT],
    key: jaxtyping.PRNGKeyArray,
    initial_parameters: ParametersT,
    config: SGLDConfig,
) -> ParametersT:
    """Run SGLD updates using ``grad_estimator``."""

    n_iters = config.num_samples
    split_keys = jrandom.split(key, 2 * n_iters)
    grad_keys = split_keys[:n_iters]
    noise_keys = split_keys[n_iters:]

    if jax.tree_util.tree_structure(config.step_size) == jax.tree_util.tree_structure(
        initial_parameters
    ):  # type: ignore[operator]
        step_sizes = config.step_size
    else:
        step_sizes = jax.tree_util.tree_map(
            lambda _: config.step_size, initial_parameters
        )

    @scan_tqdm(n_iters)
    def step(
        carry: tuple[int, ParametersT],
        inp: tuple[jaxtyping.PRNGKeyArray, jaxtyping.PRNGKeyArray],
    ):
        ix, params = carry
        _, (g_key, n_key) = inp
        grad = grad_estimator(params, g_key)
        noise = _tree_randn_like(n_key, params)
        updates = jax.tree_util.tree_map(
            lambda g, n, s: 0.5 * s * g + jnp.sqrt(s) * n,
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
        step, (0, initial_parameters), (jnp.arange(n_iters), (grad_keys, noise_keys))
    )[1]
    return samples


@inference_method
def run_full_sgld_mcmc[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        ParticleT,
        InitialParticleT,
        TransitionParticleHistoryT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
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
) -> tuple[InferenceParametersT, typing.Any]:
    score_increment = build_score_increment(target_posterior)

    particle_filter = config.particle_filter
    if hasattr(particle_filter, "proposal") and hasattr(
        particle_filter.proposal, "target_parameters"
    ):
        particle_filter = eqx.tree_at(
            lambda pf: pf.proposal.target_parameters,
            particle_filter,
            target_posterior.target_parameter,
        )

    @jax.jit
    def grad_estimator(params, key):
        model_params = target_posterior.target_parameter(params)
        out = particlefilter.run_filter(
            particle_filter,
            key,
            model_params,
            observation_path,
            recorders=(score_increment, lambda x: x.ancestor_ix),
            target_parameters=target_posterior.target_parameter,
        )

        log_weights, _, (score_increments, ancestor_ix) = out

        def accumulate_score(score, inputs):
            score_increment, ancestor_ix = inputs
            last_score = jax.tree_util.tree_map(
                lambda leaf: jax.vmap(jax.lax.dynamic_index_in_dim, in_axes=[None, 0])(
                    leaf, ancestor_ix
                ).squeeze(),
                score,
            )
            return (
                jax.tree_util.tree_map(
                    lambda *xs: sum(xs), last_score, score_increment
                ),
                None,
            )

        final_scores, _ = jax.lax.scan(
            accumulate_score,
            util.index_pytree(score_increments, 0),
            (util.slice_pytree(score_increments, 1, len(ancestor_ix)), ancestor_ix[1:]),
        )

        grad_in_target_space = jax.tree_util.tree_map(
            lambda leaf: jnp.sum(leaf * jax.nn.softmax(log_weights)),
            final_scores,
        )

        _, pullback = jax.vjp(lambda p: target_posterior.target_parameter(p), params)
        (grad_inference_space,) = pullback(grad_in_target_space)

        return grad_inference_space

    def estimate_log_joint(params, key):
        model_params = target_posterior.target_parameter(params)
        _, _, (log_marginal_increments,) = run_filter(
            particle_filter,
            key,
            model_params,
            observation_path,
            recorders=(log_marginal,),
            target_parameters=target_posterior.target_parameter,
        )
        log_prior = target_posterior.parameter_prior.log_prob(params, hyperparameters)
        return jnp.sum(log_marginal_increments) + log_prior

    init_time_start = time.time()
    init_key, sample_key = jrandom.split(key)
    initial_parameter_samples = jax.vmap(
        target_posterior.parameter_prior.sample, in_axes=[0, None]
    )(jrandom.split(init_key, config.initial_parameter_guesses), hyperparameters)
    parameter_init_marginals = jax.vmap(jax.jit(estimate_log_joint), in_axes=[0, None])(
        initial_parameter_samples,
        key,
    )
    init_time_end = time.time()
    init_time_s = init_time_end - init_time_start

    initial_parameters = util.index_pytree(
        initial_parameter_samples, jnp.argmax(parameter_init_marginals).item()
    )

    sample_time_start = time.time()
    samples = run_sgld(
        grad_estimator,
        sample_key,
        initial_parameters,
        config,
    )
    sample_time_end = time.time()
    sample_time_s = sample_time_end - sample_time_start
    time_array_s = init_time_s + (
        jnp.arange(config.num_samples) * (sample_time_s / config.num_samples)
    )

    return samples, (time_array_s,)


"""
Sketch implmentation for buffer sgld - needs checking
"""

# @inference_method
# def run_buffer_sgld_mcmc(
#     target_posterior: BayesianSequentialModel[
#         ParticleType,
#         ObservationType,
#         ConditionType,
#         ParametersType,
#         InferenceParametersType,
#         HyperParametersType,
#     ],
#     hyperparameters: HyperParametersType,
#     key: jaxtyping.PRNGKeyArray,
#     observation_path: ObservationType,
#     config: SGLDConfig,
#     *,
#     condition_path: ConditionType | None = None,
#     initial_latents: ParticleType | None = None,
#     initial_conditions: tuple[ConditionType, ...] | None = None,
#     observation_history: tuple[ObservationType, ...] | None = None,
#     test_samples: int = 1000,
# ) -> tuple[jaxtyping.Array, InferenceParametersType, typing.Any]:
#     score_increment = build_score_increment(target_posterior)
#     sequence_length = observation_path.y.shape[0]

#     def estimate_buffered_score(params, key, buffer_length=5, batch_length=10):
#         model_params = target_posterior.target_parameter(params)
#         pf_key, start_key = jrandom.split(key)
#         sample_length = 2 * buffer_length + batch_length
#         start_ix = jrandom.choice(
#             start_key, jnp.arange(sequence_length - sample_length)
#         )
#         log_weights, _, (log_marginal_increments, ancestor_ix) = (
#             particlefilter.run_filter(
#                 config.particle_filter,
#                 pf_key,
#                 model_params,
#                 util.slice_pytree(observation_path, start_ix, start_ix + sample_length),
#                 recorders=(
#                     score_increment,
#                     lambda x: x.ancestor_ix,
#                 ),
#             )
#         )

#         def accumulate_score(score, inputs):
#             score_increment, ancestor_ix, in_batch = inputs
#             last_score = jax.tree_util.tree_map(
#                 lambda leaf: jax.vmap(jax.lax.dynamic_index_in_dim, in_axes=[None, 0])(
#                     leaf, ancestor_ix
#                 ).squeeze(),
#                 score,
#             )
#             return (
#                 jax.tree_util.tree_map(
#                     lambda *xs: sum(xs),
#                     last_score,
#                     jax.tree_util.tree_map(
#                         lambda leaf: in_batch * leaf, score_increment
#                     ),
#                 ),
#                 None,
#             )

#         estimate_scale = (sequence_length - batch_length + 1) / batch_length
#         batch_mask = jnp.hstack(
#             [
#                 jnp.zeros(buffer_length),
#                 estimate_scale * jnp.ones(batch_length),
#                 jnp.zeros(buffer_length),
#             ]
#         )

#         final_scores, _ = jax.lax.scan(
#             accumulate_score,
#             jax.tree_util.tree_map(
#                 lambda x: jnp.full_like(x, fill_value=0.0),
#                 util.index_pytree(log_marginal_increments, 0),
#             ),
#             (
#                 log_marginal_increments,
#                 ancestor_ix,
#                 batch_mask,
#             ),
#         )

#         return jax.tree_util.tree_map(
#             lambda leaf: jnp.sum(leaf * jax.nn.softmax(log_weights)),
#             final_scores,
#         )

#     @jax.jit
#     def batched_estimate_score(
#         params, key, buffer_length=16, batch_length=32, batch_size=20
#     ):
#         batch_estimates = jax.vmap(
#             estimate_buffered_score, in_axes=[None, 0, None, None]
#         )(
#             params,
#             jrandom.split(key, batch_size),
#             buffer_length,
#             batch_length,
#         )
#         return jax.tree_util.tree_map(jnp.mean, batch_estimates)

#     def estimate_log_joint(params, key):
#         model_params = target_posterior.target_parameter(params)
#         _, _, (log_marginal_increments,) = run_filter(
#             config.particle_filter,
#             key,
#             model_params,
#             observation_path,
#             recorders=(log_marginal,),
#             target_parameters=target_posterior.target_parameter,
#         )
#         log_prior = target_posterior.parameter_prior.log_prob(params, hyperparameters)
#         return jnp.sum(log_marginal_increments) + log_prior

#     init_time_start = time.time()
#     init_key, sample_key = jrandom.split(key)
#     initial_parameter_samples = jax.vmap(
#         target_posterior.parameter_prior.sample, in_axes=[0, None]
#     )(jrandom.split(init_key, config.initial_parameter_guesses), hyperparameters)
#     parameter_init_marginals = jax.vmap(jax.jit(estimate_log_joint), in_axes=[0, None])(
#         initial_parameter_samples,
#         key,
#     )
#     init_time_end = time.time()
#     init_time_s = init_time_end - init_time_start

#     initial_parameters = util.index_pytree(
#         initial_parameter_samples, jnp.argmax(parameter_init_marginals).item()
#     )

#     sample_time_start = time.time()
#     samples = run_sgld(
#         batched_estimate_score,
#         sample_key,
#         initial_parameters,
#         config,
#     )
#     sample_time_end = time.time()
#     sample_time_s = sample_time_end - sample_time_start
#     time_array_s = init_time_s + (
#         jnp.arange(config.num_samples) * (sample_time_s / config.num_samples)
#     )

#     return time_array_s, samples, None
