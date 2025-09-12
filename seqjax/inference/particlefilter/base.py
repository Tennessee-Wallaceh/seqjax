from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Generic, Protocol
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
from seqjax import util
from .resampling import Resampler
from .metrics import compute_esse_from_log_weights


class FilterData(eqx.Module):
    """
    Encapsulates the data arising from a filter step
    """

    log_w: Array
    particles: Array
    ancestor_ix: Array
    observation: Array
    obs_hist: Array
    condition: Array
    last_log_w: Array
    last_particles: Array
    ess_e: Array
    log_weight_increment: Array
    parameters: Array


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
    target_parameters: Callable = lambda x: x

    def sample(
        self,
        key: PRNGKeyArray,
        particle_history: tuple[ParticleType, ...],
        observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> ParticleType:
        return self.transition.sample(
            key, particle_history, condition, self.target_parameters(parameters)
        )

    def log_prob(
        self,
        particle_history: tuple[ParticleType, ...],
        observation: ObservationType,
        particle: ParticleType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar:
        return self.transition.log_prob(
            particle_history, particle, condition, self.target_parameters(parameters)
        )


def proposal_from_transition(
    transition: Transition[ParticleType, ConditionType, ParametersType],
    target_parameters: Callable = lambda x: x,
) -> TransitionProposal[ParticleType, ObservationType, ConditionType, ParametersType]:
    return TransitionProposal(
        transition=transition,
        order=transition.order,
        target_parameters=target_parameters,
    )


class Recorder(Protocol):
    def __call__(
        self,
        log_weights: Array,
        particles: tuple[ParticleType, ...],
        ancestors: Array,
        observation: ObservationType,
        condition: ConditionType,
        last_log_weights: Array,
        last_particles: tuple[ParticleType, ...],
        ess_e: Array,
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

    def sample_step(
        self,
        step_key: PRNGKeyArray,
        log_w: Array,
        particles: tuple[ParticleType, ...],
        observation_history: tuple[ObservationType, ...],
        observation: ObservationType,
        condition: ConditionType,
        params: ParametersType,
        target_parameters: Callable = lambda x: x,
    ) -> tuple[
        Array,
        tuple[ParticleType, ...],
        tuple[ObservationType, ...],
        Scalar,
        Scalar,
        Array,
    ]:
        resample_key, proposal_key = jrandom.split(step_key)

        ess_e = compute_esse_from_log_weights(log_w)
        particles, log_w, ancestor_ix = self.resampler(
            resample_key, log_w, particles, ess_e, self.num_particles
        )

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
            self.transition_logp(
                transition_history, next_particles, condition, target_parameters(params)
            )
            + self.emission_logp(
                emission_particles,
                obs_history,
                observation,
                condition,
                target_parameters(params),
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

        return (log_w, particles, observation_history, ess_e, ancestor_ix, inc_weight)


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
    target_parameters: Callable = lambda x: x,
) -> tuple[
    Array,
    tuple[ParticleType, ...],
    tuple[PyTree, ...],
]:
    """
    Run a filtering pass over ``observation_path``.
    The first entry of observation_path corresponds to time step 0.
    Optional observation_history provides necessary history for the first evaluation.
    """

    sequence_length = jax.tree_util.tree_leaves(observation_path)[0].shape[0]

    if initial_conditions is None:
        if smc.target.prior.order > 1:
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

    if condition_path is None:
        condition_path: Any = [None] * sequence_length
    else:
        condition_path = condition_path

    init_key, *step_keys = jrandom.split(key, sequence_length)

    # Run initial step, this needs special handling because we sample from prior
    # rather than the proposal.
    init_particles = jax.vmap(smc.target.prior.sample, in_axes=[0, None, None])(
        jrandom.split(init_key, smc.num_particles),
        initial_conditions,
        target_parameters(parameters),
    )
    log_weights = smc.emission_logp(
        init_particles,
        observation_history,
        util.index_pytree(observation_path, 0),
        util.index_pytree(condition_path, 0),
        target_parameters(parameters),
    )
    if smc.target.emission.observation_dependency > 0:
        observation_history = (
            *observation_history,
            util.index_pytree(observation_path, 0),
        )

    filter_data = FilterData(
        log_w=log_weights,
        particles=init_particles,
        ancestor_ix=jnp.full((smc.num_particles,), -1, dtype=jnp.int32),
        observation=util.index_pytree(observation_path, 0),
        obs_hist=observation_history,
        condition=util.index_pytree(condition_path, 0),
        last_log_w=jnp.zeros(smc.num_particles),
        last_particles=jax.tree_util.tree_map(
            lambda x: jnp.full_like(x, fill_value=-1.0), init_particles
        ),
        ess_e=compute_esse_from_log_weights(log_weights),
        log_weight_increment=log_weights,
        parameters=parameters,
    )
    intial_record = (
        tuple(r(filter_data) for r in recorders) if recorders is not None else ()
    )

    # Define the main body
    def body(state, inputs):
        step_key, observation, condition = inputs
        last_log_w, last_particles, obs_hist = state

        log_w, particles, obs_hist, ess_e, ancestor_ix, weight_inc = smc.sample_step(
            step_key,
            last_log_w,
            last_particles,
            obs_hist,
            observation,
            condition,
            parameters,
            target_parameters,
        )
        filter_data = FilterData(
            log_w=log_w,
            particles=particles,
            ancestor_ix=ancestor_ix,
            observation=observation,
            obs_hist=obs_hist,
            condition=condition,
            last_log_w=last_log_w,
            last_particles=last_particles,
            ess_e=ess_e,
            log_weight_increment=weight_inc,
            parameters=parameters,
        )
        recorder_vals = (
            tuple(r(filter_data) for r in recorders) if recorders is not None else ()
        )
        return (
            log_w,
            particles,
            obs_hist,
        ), recorder_vals

    observation_path = util.slice_pytree(observation_path, 1, sequence_length)
    condition_path = util.slice_pytree(condition_path, 1, sequence_length)

    final_state, recorder_history = jax.lax.scan(
        body,
        (log_weights, init_particles, observation_history),
        (jnp.array(step_keys), observation_path, condition_path),
    )

    def expand_concat(value, array):
        return jnp.concatenate([jnp.expand_dims(value, axis=0), array], axis=0)

    recorder_history = jax.tree_util.tree_map(
        expand_concat, intial_record, recorder_history
    )

    log_weights, particles, _ = final_state

    return (
        log_weights,
        particles,
        recorder_history,
    )


def vmapped_run_filter(
    smc: SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType],
    key: Array,
    parameters: ParametersType,
    observation_path: Batched[ObservationType, SequenceAxis],
    condition_path: Batched[ConditionType, SequenceAxis] | None = None,
    *,
    initial_conditions: tuple[ConditionType, ...] | None = None,
    observation_history: tuple[ObservationType, ...] | None = None,
    recorders: tuple[Recorder, ...] | None = None,
) -> tuple[
    Array,
    tuple[ParticleType, ...],
    Array,
    tuple[PyTree, ...],
]:
    """Vectorise :func:`run_filter` over a leading batch dimension."""

    cond_axes = 0 if condition_path is not None else None

    def _run(key, params, obs, cond):
        return run_filter(
            smc,
            key,
            params,
            obs,
            cond,
            initial_conditions=initial_conditions,
            observation_history=observation_history,
            recorders=recorders,
        )

    run_vmap = jax.vmap(_run, in_axes=(0, 0, 0, cond_axes))

    return run_vmap(
        key,
        parameters,
        observation_path,
        condition_path,
    )
