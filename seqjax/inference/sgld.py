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
from seqjax.inference.particlefilter import SMCSampler, run_filter, log_marginal
from seqjax.inference import particlefilter
from seqjax.inference.interface import inference_method
from functools import partial
from seqjax.inference.vi.base import sample_batch_and_mask
from seqjax.inference.particlefilter import SMCSampler, run_filter
from seqjax.inference.particlefilter.base import TransitionProposal, FilterData

class SGLDConfig[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
):
    """Configuration for :func:`run_sgld`."""

    particle_filter: SMCSampler[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ]
    step_size: float | ParametersT = 1e-3
    num_samples: int = 100
    initial_parameter_guesses: int = 20


class BufferedSGLDConfig[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
):
    """Configuration for :func:`run_sgld`."""

    particle_filter: SMCSampler[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
    ]
    step_size: float | ParametersT = 1e-3
    num_samples: int = 100
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

def estimate_initial_step_score(model, filter_data: FilterData):
    # grad wrt to parameters
    def model_initial_log_prob(particles, inference_parameters):    
        model_parameters = model.target_parameter(inference_parameters)
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
        model_parameters = model.target_parameter(inference_parameters)
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
    # associated with the ancestor index, we must
    # respect the histories whilst accumulating
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
    noise_rescale: float = 1.
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
        step, (0, initial_parameters), (jnp.arange(n_iters), (grad_keys, noise_keys))
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

    def _buffered_estimate_score(particle_filter, observation_path, model, batch_length, buffer_length, params, key):
        latent_scaling = (batch_length + observation_path.batch_shape[0] - 1) / batch_length

        batch_key, pf_key = jrandom.split(key)
        approx_start, y_batch, c_batch, theta_mask = sample_batch_and_mask(
            batch_key, 
            sequence_length,
            batch_length=batch_length,
            buffer_length=buffer_length,
            y_path=observation_path,
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
            target_parameters=model.target_parameter,
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
        log_prior_score = jax.grad(model.parameter_prior.log_prob, argnums=0)(params, None)
        # use rescaling in Aicher code
        return jax.tree_util.tree_map(
            lambda *x: sum(x) / sequence_length,  log_prior_score, final_score
        )

    score_esimator = jax.jit(partial(
        _buffered_estimate_score,
        config.particle_filter, 
        observation_path, 
        target_posterior,
        config.batch_length,
        config.buffer_length,
    ))

    init_time_start = time.time()
    init_key, sample_key = jrandom.split(key)
    initial_parameters = target_posterior.parameter_prior.sample(init_key, None)
    init_time_end = time.time()
    init_time_s = init_time_end - init_time_start

    sample_time_start = time.time()
    samples = run_sgld(
        score_esimator,
        sample_key,
        initial_parameters,
        config,
        noise_rescale=1 / sequence_length # this is what Aicher does 
    )
    sample_time_end = time.time()
    sample_time_s = sample_time_end - sample_time_start
    time_array_s = init_time_s + (
        jnp.arange(config.num_samples) * (sample_time_s / config.num_samples)
    )

    return samples, None
