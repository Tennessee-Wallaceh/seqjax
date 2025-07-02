from __future__ import annotations

from typing import Callable, Protocol, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray

from seqjax.model.base import Target, ParticleType, ObservationType, ConditionType, ParametersType, Transition
from seqjax.util import dynamic_index_pytree_in_dim as index_tree

def resample_from_log_weights(key, log_weights, particles):
    # gumbel max trick
    gumbels = -jnp.log(-jnp.log(
        jrandom.uniform(key, (log_weights.shape[0], log_weights.shape[0]))
    ))
    particle_ix = jnp.argmax(log_weights + gumbels, axis=1).reshape(-1)
    return jax.vmap(
        index_tree, in_axes=[0, None],
    )(particle_ix, particles)

def compute_esse_from_log_weights(log_weights):
    # ess efficiency, ie ess / M
    log_w = log_weights - jnp.max(log_weights)  # for numerical stability
    log_sum_w = jax.scipy.special.logsumexp(log_weights)
    log_sum_w2 = jax.scipy.special.logsumexp(2 * log_weights)
    ess = jnp.exp(2 * log_sum_w - log_sum_w2)
    return ess / log_w.shape[0]

class SampleStep(Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        log_weights: Array,
        particles: ParticleType,
        observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> tuple[Array, ParticleType, Float[Array, ""]]:
        ...


class GeneralSequentialImportanceSampler(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType],
):
    """Base class implementing sequential importance sampling."""

    target: Target[ParticleType, ObservationType, ConditionType, ParametersType]
    proposal: Transition[ParticleType, ConditionType, ParametersType]
    resampler: Callable[[PRNGKeyArray, Array, ParticleType], ParticleType]

    def __init__(
        self,
        target: Target[ParticleType, ObservationType, ConditionType, ParametersType],
        proposal: Transition[ParticleType, ConditionType, ParametersType],
        resampler: Callable[[PRNGKeyArray, Array, ParticleType], ParticleType] = resample_from_log_weights,
    ) -> None:
        super().__init__()
        self.target = target
        self.proposal = proposal
        self.resampler = resampler

    # ------------------------------------------------------------------
    def configure_filter(
        self,
        num_particles: int,
        key: PRNGKeyArray,
        parameters: ParametersType,
        initial_conditions: tuple[ConditionType, ...] | None = None,
    ) -> tuple[Array, ParticleType, SampleStep]:
        if initial_conditions is None:
            initial_conditions = tuple(None for _ in range(self.target.prior.order))

        init_particles = jax.vmap(self.target.prior.sample, in_axes=[0, None, None])(
            jrandom.split(key, num_particles),
            initial_conditions,
            parameters,
        )

        log_weights = jnp.zeros((num_particles,))

        proposal_sample = jax.vmap(self.proposal.sample, in_axes=[0, 0, None, None])
        proposal_logp = jax.vmap(self.proposal.log_p, in_axes=[0, 0, None, None])
        transition_logp = jax.vmap(
            self.target.transition.log_p, in_axes=[0, 0, None, None]
        )
        emission_logp = jax.vmap(
            self.target.emission.log_p, in_axes=[0, None, None, None]
        )
        resample = self.resampler

        def sample_step(
            step_key: PRNGKeyArray,
            log_w: Array,
            particles: ParticleType,
            observation: ObservationType,
            condition: ConditionType,
            params: ParametersType,
        ) -> tuple[Array, ParticleType, Float[Array, ""]]:
            resample_key, proposal_key = jrandom.split(step_key)
            ess_e = compute_esse_from_log_weights(log_w)
            particles = jax.lax.cond(
                ess_e < 0.5,
                lambda p: resample(resample_key, log_w, p),
                lambda p: p,
                particles,
            )
            log_w = jax.lax.select(ess_e < 0.5, jnp.zeros_like(log_w), log_w)

            next_particles = proposal_sample(
                jrandom.split(proposal_key, num_particles),
                particles,
                condition,
                params,
            )

            inc_weight = (
                transition_logp(particles, next_particles, condition, params)
                + emission_logp(next_particles, observation, condition, params)
                - proposal_logp(particles, next_particles, condition, params)
            )
            log_w = log_w + inc_weight

            return log_w, next_particles, ess_e

        return log_weights, init_particles, sample_step


class BootstrapParticleFilter(
    GeneralSequentialImportanceSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
):
    """Classical bootstrap particle filter."""

    def __init__(self, target: Target[ParticleType, ObservationType, ConditionType, ParametersType]) -> None:
        super().__init__(target, target.transition, resample_from_log_weights)

