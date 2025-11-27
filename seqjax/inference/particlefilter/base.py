import typing
from functools import cached_property
from typing import Callable, Protocol
from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
from seqjax.model.base import (
    SequentialModel,
    Transition,
)
import seqjax.model.typing as seqjtyping
from seqjax import util
from .resampling import Resampler
from .metrics import compute_esse_from_log_weights


@dataclass
class FilterData:
    """
    Encapsulates the data arising from a filter step
    """

    log_w: Array
    particles: tuple[seqjtyping.Latent, ...]
    ancestor_ix: Array
    observation: seqjtyping.Observation
    observation_history: tuple[seqjtyping.Observation, ...]
    condition: seqjtyping.Condition | seqjtyping.NoCondition
    last_log_w: Array
    last_particles: Array
    ess_e: Array
    log_weight_increment: Array
    parameters: seqjtyping.Parameters


class Recorder(Protocol):
    """
    The filtering recorder is just a function producing some output from the current filter data.
    """

    def __call__(self, filter_data: FilterData) -> typing.Any: ...


class Proposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
):
    """
    Proposal distribution for sequential importance sampling.
    Implementation is via an eqx.Module to support parameterized proposals.
    The proposal can maintain a longer particle history than required for the model.

    """

    order: eqx.AbstractClassVar[int]

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[ParticleT, ...],
        observation: ObservationT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ParticleT: ...

    @staticmethod
    @abstractmethod
    def log_prob(
        particle_history: tuple[ParticleT, ...],
        observation: ObservationT,
        particle: ParticleT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...


class TransitionProposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    HistoryT,
](
    eqx.Module,
):
    """Adapter converting a ``Transition`` to a ``Proposal``."""

    transition: Transition[ParticleT, ConditionT, ParametersT, HistoryT]
    order: int
    # I don't actually care what this does, as long as it produces ParametersT.
    target_parameters: Callable[[ParametersT], ParametersT] = lambda x: x

    def sample(
        self,
        key: PRNGKeyArray,
        particle_history: HistoryT,
        observation: ObservationT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ParticleT:
        return self.transition.sample(
            key, particle_history, condition, self.target_parameters(parameters)
        )

    def log_prob(
        self,
        particle_history: HistoryT,
        observation: ObservationT,
        particle: ParticleT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar:
        return self.transition.log_prob(
            particle_history, particle, condition, self.target_parameters(parameters)
        )


def proposal_from_transition[
    ParticleT: seqjtyping.Latent,
    TransitionParticleHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    transition: Transition[
        TransitionParticleHistoryT, ParticleT, ConditionT, ParametersT
    ],
    target_parameters: Callable = lambda x: x,
) -> TransitionProposal[
    ParticleT, TransitionParticleHistoryT, ObservationT, ConditionT, ParametersT
]:
    return TransitionProposal(
        transition=transition,
        order=transition.order,
        target_parameters=target_parameters,
    )


class SMCSampler[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    eqx.Module,
):
    """Base class implementing sequential Monte Carlo."""

    target: SequentialModel[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
    ]
    proposal: Proposal[ParticleT, ObservationT, ConditionT, ParametersT]
    resampler: Resampler
    num_particles: int

    @cached_property
    def proposal_sample(self) -> Callable:
        return jax.vmap(self.proposal.sample, in_axes=[0, 0, None, None, None])

    @cached_property
    def proposal_log_prob(self) -> Callable:
        return jax.vmap(self.proposal.log_prob, in_axes=[0, None, 0, None, None])

    @cached_property
    def transition_log_prob(self) -> Callable:
        return jax.vmap(self.target.transition.log_prob, in_axes=[0, 0, None, None])

    @cached_property
    def emission_log_prob(self) -> Callable:
        return jax.vmap(
            self.target.emission.log_prob,
            in_axes=[0, None, None, None, None],
        )

    def sample_step(
        self,
        step_key: PRNGKeyArray,
        log_w: Array,
        particles: tuple[ParticleT, ...],
        observation_history: tuple[ObservationT, ...],
        observation: ObservationT,
        condition: ConditionT,
        params: ParametersT,
        target_parameters: Callable = lambda x: x,
    ) -> tuple[
        Array,
        tuple[ParticleT, ...],
        tuple[ObservationT, ...],
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


def run_filter[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](
    smc: SMCSampler[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    key: PRNGKeyArray,
    parameters: ParametersT,
    observation_path: ObservationT,
    condition_path: ConditionT,
    *,
    recorders: tuple[Recorder, ...] | None = None,
    target_parameters: Callable = lambda x: x,
) -> tuple[
    Array,
    ParticleT,
    tuple[PyTree, ...],
]:
    """
    Run a filtering pass over ``observation_path``.
    The first entry of observation_path corresponds to time step 0.
    Optional observation_history provides necessary history for the first evaluation.
    """

    sequence_length = jax.tree_util.tree_leaves(observation_path)[0].shape[0]

    # TODO: validate initial conditions and observation history
    # if initial_conditions is None:
    #     if smc.target.prior.order > 1:
    #         raise ValueError(
    #             "initial_conditions must be provided when the prior has order > 0"
    #         )
    initial_conditions = ()

    # if observation_history is None:
    #     if smc.target.emission.observation_dependency > 0:
    #         raise ValueError(
    #             "observation_history must be provided when the emission has observation dependency > 0"
    #         )
    observation_history = ()

    if condition_path is None:
        initial_condition = None
    else:
        initial_condition = util.index_pytree(condition_path, 0)

    init_key, *step_keys = jrandom.split(key, sequence_length)

    # Run initial step, this needs special handling because we sample from prior
    # rather than the proposal.
    init_particles = jax.vmap(smc.target.prior.sample, in_axes=[0, None, None])(
        jrandom.split(init_key, smc.num_particles),
        typing.cast(typing.Any, initial_conditions),
        target_parameters(parameters),
    )
    log_weights = smc.emission_logp(
        init_particles,
        observation_history,
        util.index_pytree(observation_path, 0),
        initial_condition,
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
        condition=initial_condition,
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
        if sliced_condition_path is None:
            step_key, observation = inputs
            condition = None
        else:
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
    if condition_path is not None:
        sliced_condition_path = util.slice_pytree(condition_path, 1, sequence_length)
    else:
        sliced_condition_path = None

    body_inputs: tuple[typing.Any, ...]
    if sliced_condition_path is None:
        body_inputs = (jnp.array(step_keys), observation_path)
    else:
        body_inputs = (
            jnp.array(step_keys),
            observation_path,
            sliced_condition_path,
        )

    final_state, recorder_history = jax.lax.scan(
        body,
        (log_weights, init_particles, observation_history),
        body_inputs,
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
