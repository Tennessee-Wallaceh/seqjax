from __future__ import annotations

from typing import Callable, Protocol, Generic
from functools import partial, cached_property

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from seqjax.model.base import (
    Target,
    ParticleType,
    ObservationType,
    ConditionType,
    ParametersType,
    Transition,
)
from seqjax.util import dynamic_index_pytree_in_dim as index_tree

Resampler = Callable[[PRNGKeyArray, Array, ParticleType, Scalar], ParticleType]


def gumbel_resample_from_log_weights(key, log_weights, particles, ess_e):
    # gumbel max trick
    gumbels = -jnp.log(
        -jnp.log(jrandom.uniform(key, (log_weights.shape[0], log_weights.shape[0])))
    )
    particle_ix = jnp.argmax(log_weights + gumbels, axis=1).reshape(-1)
    return jax.vmap(index_tree, in_axes=[None, 0, None])(particles, particle_ix, 0)


def conditional_resample(
    key, log_weights, particles, ess_e, resampler: Resampler, esse_threshold: float
):
    particles = jax.lax.cond(
        ess_e < esse_threshold,
        lambda p: resampler(key, log_weights, p, ess_e),
        lambda p: p,
        particles,
    )

    return particles


def compute_esse_from_log_weights(log_weights: Float[Array, "num_particles"]) -> Scalar:
    # ess efficiency, ie ess / M
    log_w = log_weights - jnp.max(log_weights)  # for numerical stability
    log_sum_w = jax.scipy.special.logsumexp(log_weights)
    log_sum_w2 = jax.scipy.special.logsumexp(2 * log_weights)
    ess = jnp.exp(2 * log_sum_w - log_sum_w2)
    return ess / log_w.shape[0]


class GeneralSequentialImportanceSampler(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType],
):
    """Base class implementing sequential importance sampling."""

    target: Target[ParticleType, ObservationType, ConditionType, ParametersType]
    proposal: Transition[ParticleType, ConditionType, ParametersType]
    resampler: Callable[[PRNGKeyArray, Array, ParticleType, Scalar], ParticleType]
    num_particles: int

    # Vectorised helpers are exposed via cached properties to avoid repeated
    # recompilation whilst keeping the module definition declarative.

    @cached_property
    def proposal_sample(self) -> Callable:
        return jax.vmap(self.proposal.sample, in_axes=[0, 0, None, None])

    @cached_property
    def proposal_logp(self) -> Callable:
        return jax.vmap(self.proposal.log_p, in_axes=[0, 0, None, None])

    @cached_property
    def transition_logp(self) -> Callable:
        return jax.vmap(self.target.transition.log_p, in_axes=[0, 0, None, None])

    @cached_property
    def emission_logp(self) -> Callable:
        return jax.vmap(
            lambda p, o, c, r: self.target.emission.log_p(p, (), o, c, r),
            in_axes=[0, None, None, None],
        )

    def sample_step(
        self,
        step_key: PRNGKeyArray,
        log_w: Array,
        particles: tuple[ParticleType, ...],
        observation: ObservationType,
        condition: ConditionType,
        params: ParametersType,
    ) -> tuple[Array, tuple[ParticleType, ...], Float[Array, ""]]:
        """Advance the filter by one step and maintain particle history."""

        resample_key, proposal_key = jrandom.split(step_key)
        ess_e = compute_esse_from_log_weights(log_w)

        # Resample complete particle histories if necessary
        particles = self.resampler(resample_key, log_w, particles, ess_e)

        # Determine which part of the history the proposal/transition depend on
        proposal_history = particles[-self.proposal.order :]
        transition_history = particles[-self.target.transition.order :]

        next_particles = self.proposal_sample(
            jrandom.split(proposal_key, self.num_particles),
            proposal_history,
            condition,
            params,
        )

        emission_history = particles[-(self.target.emission.order - 1) :] if self.target.emission.order > 1 else ()
        emission_particles = (*emission_history, next_particles)

        inc_weight = (
            self.transition_logp(transition_history, next_particles, condition, params)
            + self.emission_logp(emission_particles, observation, condition, params)
            - self.proposal_logp(proposal_history, next_particles, condition, params)
        )
        log_w = log_w + inc_weight

        max_order = max(self.target.transition.order, self.target.emission.order)
        particles = (*particles, next_particles)[-max_order:]

        return log_w, particles, ess_e


def run_filter(
    gsis: GeneralSequentialImportanceSampler[ParticleType, ObservationType, ConditionType, ParametersType],
    key: PRNGKeyArray,
    parameters: ParametersType,
    observation_path,
    condition_path=None,
    initial_conditions: tuple[ConditionType, ...] | None = None,
) -> tuple[Array, tuple[ParticleType, ...], Array]:
    """Run a full filtering pass over ``observation_path``.

    Parameters
    ----------
    gsis:
        The sequential importance sampler to run.
    key:
        PRNG key used for all sampling steps.
    parameters:
        Model parameters.
    observation_path:
        Sequence of observations to filter.
    condition_path:
        Optional sequence of conditions. ``None`` broadcasts ``None`` for all
        steps.
    initial_conditions:
        Conditions for sampling the initial particles. ``None`` results in a
        tuple of ``None`` of appropriate length.

    Returns
    -------
    log_weights:
        Log weights of particles after processing the full sequence.
    particles:
        Particle history after the final step.
    ess_history:
        Effective sample size efficiency for each step.
    """

    sequence_length = jax.tree_util.tree_leaves(observation_path)[0].shape[0]

    if initial_conditions is None:
        initial_conditions = tuple(None for _ in range(gsis.target.prior.order))

    init_key, *step_keys = jrandom.split(key, sequence_length + 1)

    init_particles = jax.vmap(gsis.target.prior.sample, in_axes=[0, None, None])(
        jrandom.split(init_key, gsis.num_particles),
        initial_conditions,
        parameters,
    )

    log_weights = jnp.zeros((gsis.num_particles,))

    def body(state, inputs):
        step_key, observation, condition = inputs
        log_w, particles = state
        log_w, particles, ess_e = gsis.sample_step(
            step_key, log_w, particles, observation, condition, parameters
        )
        return (log_w, particles), ess_e

    if condition_path is None:
        cond_seq = [None] * sequence_length
    else:
        cond_seq = condition_path

    final_state, ess_history = jax.lax.scan(
        body,
        (log_weights, init_particles),
        (jnp.array(step_keys), observation_path, cond_seq),
    )

    log_weights, particles = final_state

    return log_weights, particles, ess_history

class BootstrapParticleFilter(
    GeneralSequentialImportanceSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
):
    """Classical bootstrap particle filter."""

    def __init__(
        self,
        target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
        num_particles: int,
    ) -> None:
        super().__init__(
            target=target,
            proposal=target.transition,
            resampler=partial(
                conditional_resample,
                resampler=gumbel_resample_from_log_weights,
                esse_threshold=0.5,
            ),
            num_particles=num_particles,
        )
