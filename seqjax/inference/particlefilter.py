from __future__ import annotations

from functools import partial
from typing import Callable, Generic

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

Resampler = Callable[[PRNGKeyArray, Array, ParticleType, float], ParticleType]


def gumbel_resample_from_log_weights(key, log_weights, particles, ess_e):
    # gumbel max trick
    gumbels = -jnp.log(
        -jnp.log(jrandom.uniform(key, (log_weights.shape[0], log_weights.shape[0])))
    )
    particle_ix = jnp.argmax(log_weights + gumbels, axis=1).reshape(-1)
    return jax.vmap(
        index_tree,
        in_axes=[0, None],
    )(particle_ix, particles)


def conditional_resample(
    key, log_weights, particles, ess_e, resampler: Resampler, esse_threshold: float
):
    particles = jax.lax.cond(
        ess_e < esse_threshold,
        lambda p: resampler(key, log_weights, p, ess_e),
        lambda p: p,
        particles,
    )


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
    resampler: Callable[[PRNGKeyArray, Array, ParticleType], ParticleType]
    num_particles: int

    def __init__(
        self,
        target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
        num_particles: int,
        proposal: Transition[ParticleType, ConditionType, ParametersType],
        resampler: Resampler = resample_from_log_weights,
    ) -> None:
        super().__init__()
        self.num_particles = num_particles
        self.target = target
        self.proposal = proposal
        self.resample = resampler
        self.proposal_sample = jax.vmap(
            self.proposal.sample, in_axes=[0, 0, None, None]
        )
        self.proposal_logp = jax.vmap(self.proposal.log_p, in_axes=[0, 0, None, None])
        self.transition_logp = jax.vmap(
            self.target.transition.log_p, in_axes=[0, 0, None, None]
        )
        self.emission_logp = jax.vmap(
            self.target.emission.log_p, in_axes=[0, None, None, None]
        )

    def sample_step(
        self,
        step_key: PRNGKeyArray,
        log_w: Array,
        particles: ParticleType,
        observation: ObservationType,
        condition: ConditionType,
        params: ParametersType,
    ) -> tuple[Array, ParticleType, Float[Array, ""]]:
        resample_key, proposal_key = jrandom.split(step_key)
        ess_e = compute_esse_from_log_weights(log_w)
        particles, log_w = self.resample(resample_key, log_w, p)
        next_particles = self.proposal_sample(
            jrandom.split(proposal_key, num_particles),
            particles,
            condition,
            params,
        )

        inc_weight = (
            self.transition_logp(particles, next_particles, condition, params)
            + self.emission_logp(next_particles, observation, condition, params)
            - self.proposal_logp(particles, next_particles, condition, params)
        )
        log_w = log_w + inc_weight

        return log_w, next_particles, ess_e


def run_filter(
    gsis: GeneralSequentialImportanceSampler,
    key: PRNGKeyArray,
    parameters: ParametersType,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_path,
    condition_path,
    parameters,
) -> tuple[Array, ParticleType, SampleStep]:
    if initial_conditions is None:
        initial_conditions = tuple(None for _ in range(gsis.target.prior.order))

    init_particles = jax.vmap(gsis.target.prior.sample, in_axes=[0, None, None])(
        jrandom.split(key, gsis.num_particles),
        initial_conditions,
        parameters,
    )

    log_weights = jnp.zeros((gsis.num_particles,))

    #TODO: run scan down observations
    # return array of ess, some compression of particles?

class BootstrapParticleFilter(
    GeneralSequentialImportanceSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
):
    """Classical bootstrap particle filter."""

    def __init__(
        self,
        target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
    ) -> None:
        super().__init__(
            target,
            target.transition,
            partial(
                conditional_resample,
                resampler=gumbel_resample_from_log_weights,
                ess_thresh=0.5
            )
        )
