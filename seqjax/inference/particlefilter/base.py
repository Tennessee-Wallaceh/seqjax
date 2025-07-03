from __future__ import annotations

from functools import cached_property
from typing import Callable, Generic, Protocol

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
from seqjax.model.typing import Batched, SequenceAxis

from .resampling import Resampler
from .metrics import compute_esse_from_log_weights


class Recorder(Protocol):
    def __call__(
        self, weights: Array, particles: tuple[ParticleType, ...]
    ) -> PyTree: ...


class GeneralSequentialImportanceSampler(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType],
):
    """Base class implementing sequential importance sampling."""

    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
    proposal: Transition[ParticleType, ConditionType, ParametersType]
    resampler: Resampler
    num_particles: int

    @cached_property
    def proposal_sample(self) -> Callable:
        return jax.vmap(self.proposal.sample, in_axes=[0, 0, None, None])

    @cached_property
    def proposal_logp(self) -> Callable:
        return jax.vmap(self.proposal.log_prob, in_axes=[0, 0, None, None])

    @cached_property
    def transition_logp(self) -> Callable:
        return jax.vmap(self.target.transition.log_prob, in_axes=[0, 0, None, None])

    @cached_property
    def emission_logp(self) -> Callable:
        return jax.vmap(
            lambda p, o, c, r: self.target.emission.log_prob(p, (), o, c, r),
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
    ) -> tuple[Array, tuple[ParticleType, ...], Scalar]:
        """Advance the filter by one step and maintain particle history."""

        resample_key, proposal_key = jrandom.split(step_key)
        ess_e = compute_esse_from_log_weights(log_w)

        particles = self.resampler(resample_key, log_w, particles, ess_e)

        proposal_history = particles[-self.proposal.order :]
        transition_history = particles[-self.target.transition.order :]

        next_particles = self.proposal_sample(
            jrandom.split(proposal_key, self.num_particles),
            proposal_history,
            condition,
            params,
        )

        emission_history = (
            particles[-(self.target.emission.order - 1) :]
            if self.target.emission.order > 1
            else ()
        )
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
    gsis: GeneralSequentialImportanceSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key: PRNGKeyArray,
    parameters: ParametersType,
    observation_path: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    *,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    recorders: tuple[Recorder, ...] | None = None,
) -> tuple[Array, tuple[ParticleType, ...], Array, tuple[PyTree, ...]]:
    """Run a filtering pass over ``observation_path``."""

    sequence_length = jax.tree_util.tree_leaves(observation_path)[0].shape[0]

    if initial_conditions is None:
        if gsis.target.prior.order > 0:
            raise ValueError(
                "initial_conditions must be provided when the prior has order > 0"
            )
        initial_conditions = ()

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
        weights = jax.nn.softmax(log_w)
        recorder_vals = (
            tuple(r(weights, particles) for r in recorders)
            if recorders is not None
            else ()
        )
        return (log_w, particles), (ess_e, *recorder_vals)

    if condition_path is None:
        cond_seq = [None] * sequence_length
    else:
        cond_seq = condition_path

    final_state, scan_hist = jax.lax.scan(
        body,
        (log_weights, init_particles),
        (jnp.array(step_keys), observation_path, cond_seq),
    )

    log_weights, particles = final_state

    ess_history = scan_hist[0]
    recorder_history = tuple(scan_hist[1:])

    return log_weights, particles, ess_history, recorder_history
