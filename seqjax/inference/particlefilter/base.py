from __future__ import annotations

from functools import cached_property
from typing import Callable, Generic, Protocol
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
    Transition,
)
from seqjax.model.typing import Batched, SequenceAxis, EnforceInterface
from .resampling import Resampler
from .metrics import compute_esse_from_log_weights


class Proposal(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType],
    EnforceInterface,
):
    """Proposal distribution for sequential importance sampling."""

    order: eqx.AbstractClassVar[int]

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[ParticleType, ...],
        observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> ParticleType: ...

    @staticmethod
    @abstractmethod
    def log_prob(
        particle_history: tuple[ParticleType, ...],
        observation: ObservationType,
        particle: ParticleType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar: ...


class TransitionProposal(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType],
):
    """Adapter converting a ``Transition`` to a ``Proposal``."""

    transition: Transition[ParticleType, ConditionType, ParametersType]
    order: int

    def sample(
        self,
        key: PRNGKeyArray,
        particle_history: tuple[ParticleType, ...],
        observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> ParticleType:
        return self.transition.sample(key, particle_history, condition, parameters)

    def log_prob(
        self,
        particle_history: tuple[ParticleType, ...],
        observation: ObservationType,
        particle: ParticleType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar:
        return self.transition.log_prob(
            particle_history, particle, condition, parameters
        )


def proposal_from_transition(
    transition: Transition[ParticleType, ConditionType, ParametersType],
) -> TransitionProposal[ParticleType, ObservationType, ConditionType, ParametersType]:
    return TransitionProposal(transition=transition, order=transition.order)


class Recorder(Protocol):
    def __call__(
        self, weights: Array, particles: tuple[ParticleType, ...]
    ) -> PyTree: ...


class SMCSampler(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType],
):
    """Base class implementing sequential Monte Carlo."""

    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
    proposal: Proposal[ParticleType, ObservationType, ConditionType, ParametersType]
    resampler: Resampler
    num_particles: int

    @cached_property
    def proposal_sample(self) -> Callable:
        return jax.vmap(self.proposal.sample, in_axes=[0, 0, None, None, None])

    @cached_property
    def proposal_logp(self) -> Callable:
        return jax.vmap(self.proposal.log_prob, in_axes=[0, None, 0, None, None])

    @cached_property
    def transition_logp(self) -> Callable:
        return jax.vmap(self.target.transition.log_prob, in_axes=[0, 0, None, None])

    @cached_property
    def emission_logp(self) -> Callable:
        return jax.vmap(
            self.target.emission.log_prob,
            in_axes=[0, None, None, None, None],
        )

    def _resample_log_weights(
        self,
        log_w: Array,
        particles: tuple[ParticleType, ...],
        observation_history: tuple[ObservationType, ...],
        observation: ObservationType,
        condition: ConditionType,
        params: ParametersType,
    ) -> Array:
        """Return the log-weights used for resampling."""

        return log_w

    def sample_step(
        self,
        step_key: PRNGKeyArray,
        log_w: Array,
        particles: tuple[ParticleType, ...],
        observation_history: tuple[ObservationType, ...],
        observation: ObservationType,
        condition: ConditionType,
        params: ParametersType,
    ) -> tuple[Array, tuple[ParticleType, ...], tuple[ObservationType, ...], Scalar]:
        """Advance the filter by one step and maintain particle history."""

        resample_key, proposal_key = jrandom.split(step_key)

        log_w_resample = self._resample_log_weights(
            log_w,
            particles,
            observation_history,
            observation,
            condition,
            params,
        )

        ess_e = compute_esse_from_log_weights(log_w_resample)


        particles, log_w = self.resampler(resample_key, log_w, particles, ess_e)

        proposal_history = particles[-self.proposal.order :]
        transition_history = particles[-self.target.transition.order :]

        next_particles = self.proposal_sample(
            jrandom.split(proposal_key, self.num_particles),
            proposal_history,
            observation,
            condition,
            params,
        )

        emission_history = (
            particles[-(self.target.emission.order - 1) :]
            if self.target.emission.order > 1
            else ()
        )
        emission_particles = (*emission_history, next_particles)

        obs_history = (
            observation_history[-self.target.emission.observation_dependency :]
            if self.target.emission.observation_dependency > 0
            else ()
        )

        inc_weight = (
            self.transition_logp(transition_history, next_particles, condition, params)
            + self.emission_logp(
                emission_particles,
                obs_history,
                observation,
                condition,
                params,
            )
            - self.proposal_logp(
                proposal_history, observation, next_particles, condition, params
            )
        )
        log_w = log_w + inc_weight

        max_order = max(self.target.transition.order, self.target.emission.order)
        particles = (*particles, next_particles)[-max_order:]

        if self.target.emission.observation_dependency > 0:
            observation_history = (*observation_history, observation)[
                -self.target.emission.observation_dependency :
            ]
        else:
            observation_history = ()

        return log_w, particles, observation_history, ess_e


def run_filter(
    smc: SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType],
    key: PRNGKeyArray,
    parameters: ParametersType,
    observation_path: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    *,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_history: tuple[ObservationType, ...] | None = None,
    recorders: tuple[Recorder, ...] | None = None,
) -> tuple[Array, tuple[ParticleType, ...], Array, tuple[PyTree, ...]]:
    """Run a filtering pass over ``observation_path``."""

    sequence_length = jax.tree_util.tree_leaves(observation_path)[0].shape[0]

    if initial_conditions is None:
        if smc.target.prior.order > 0:
            raise ValueError(
                "initial_conditions must be provided when the prior has order > 0"
            )
        initial_conditions = ()

    if observation_history is None:
        if smc.target.emission.observation_dependency > 0:
            raise ValueError(
                "observation_history must be provided when the emission has observation dependency > 0"
            )
        observation_history = ()

    init_key, *step_keys = jrandom.split(key, sequence_length + 1)

    init_particles = jax.vmap(smc.target.prior.sample, in_axes=[0, None, None])(
        jrandom.split(init_key, smc.num_particles),
        initial_conditions,
        parameters,
    )

    log_weights = jnp.full((smc.num_particles,), -jnp.log(smc.num_particles))

    def body(state, inputs):
        step_key, observation, condition = inputs
        log_w, particles, obs_hist, log_mp_prev = state
        log_w, particles, obs_hist, ess_e = smc.sample_step(
            step_key,
            log_w,
            particles,
            obs_hist,
            observation,
            condition,
            parameters,
        )
        log_sum_w = jax.scipy.special.logsumexp(log_w)
        log_mp = log_mp_prev + log_sum_w - jnp.log(smc.num_particles)
        log_w = log_w - log_sum_w
        weights = jax.nn.softmax(log_w)
        recorder_vals = (
            tuple(r(weights, particles) for r in recorders)
            if recorders is not None
            else ()
        )
        return (log_w, particles, obs_hist, log_mp), (log_mp, ess_e, *recorder_vals)

    if condition_path is None:
        cond_seq = [None] * sequence_length
    else:
        cond_seq = condition_path

    final_state, scan_hist = jax.lax.scan(
        body,
        (log_weights, init_particles, observation_history, jnp.array(0.0)),
        (jnp.array(step_keys), observation_path, cond_seq),
    )

    log_weights, particles, _, log_marginal_final = final_state

    log_marginal_history = scan_hist[0]
    ess_history = scan_hist[1]
    recorder_history = tuple(scan_hist[2:])

    return (
        log_weights,
        particles,
        log_marginal_history,
        ess_history,
        recorder_history,
    )

