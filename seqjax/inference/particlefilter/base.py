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
from seqjax.model import BayesianSequentialModel
from seqjax import util
from .resampling import Resampler
from seqjax.model.evaluate import (
    slice_emission_observation_history,
    slice_prior_conditions,
)


@dataclass
class FilterData:
    """
    Encapsulates the data arising from a filter step
    """

    start_log_w: Array
    resampled_log_w: Array
    log_w: Array

    particles: tuple[seqjtyping.Latent, ...]
    ancestor_ix: Array
    log_w_inc: Array


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
    Proposal distribution for SMC.
    Implementation is via an eqx.Module to support parameterized proposals.
    The proposal can maintain a longer particle history than required for the model.
    The proposal operates on the full particle set, rather than element wise.
    This is necessary to support resampling procedures.
    """

    order: int

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[ParticleT, ...],
        observation: ObservationT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> tuple[ParticleT, Array]: ...

    """
    Leading axis is num particles
    """

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
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
    PriorLatentT: tuple[seqjtyping.Latent, ...],
    PriorConditionT: tuple[seqjtyping.Condition, ...],
    TransitionLatentHistoryT: tuple[seqjtyping.Latent, ...],
    EmissionLatentHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationHistoryT = tuple[seqjtyping.Observation, ...],
](
    Proposal,
):
    """
    Wraps a SSM ``Transition`` to a ``Proposal``.
    This is what is done to produce the "Bootstrap" particle filter.
    We also supply an optional resampling scheme.
    If there is no resampling scheme this is SIS.
    """

    transition: Transition[TransitionLatentHistoryT, ParticleT, ConditionT, ParametersT]
    # I don't actually care what this does, as long as it produces ParametersT.
    target_parameters: Callable[[InferenceParametersT], ParametersT]

    def __init__(
        self,
        model: BayesianSequentialModel[
            ParticleT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
            PriorLatentT,
            PriorConditionT,
            TransitionLatentHistoryT,
            EmissionLatentHistoryT,
            ObservationHistoryT,
        ],
    ):
        self.transition = model.target.transition
        self.target_parameters = model.target_parameter
        super().__init__(order=self.transition.order)

    def sample(
        self,
        key: PRNGKeyArray,
        particle_history: TransitionLatentHistoryT,
        observation: ObservationT,
        condition: ConditionT,
        parameters: InferenceParametersT,
    ) -> ParticleT:
        return self.transition.sample(
            key, particle_history, condition, self.target_parameters(parameters)
        )

    def log_prob(
        self,
        particle_history: TransitionLatentHistoryT,
        observation: ObservationT,
        new_particles: ParticleT,
        condition: ConditionT,
        parameters: InferenceParametersT,
    ) -> Array:
        return self.transition.log_prob(
            particle_history,
            new_particles,
            condition,
            self.target_parameters(parameters),
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
    ) -> FilterData:
        resample_key, proposal_key = jrandom.split(step_key)

        resampled_particles, ancestor_ix, resampled_log_w, _, _ = self.resampler(
            resample_key,
            log_w,
            particles,
            self.num_particles,
        )

        proposal_history = resampled_particles[-self.proposal.order :]
        transition_history = self.target.latent_view_for_transition(resampled_particles)

        proposed_particles = self.proposal_sample(
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
        emission_particles = (*emission_history, proposed_particles)

        obs_history = (
            observation_history[-self.target.emission.observation_dependency :]
            if self.target.emission.observation_dependency > 0
            else ()
        )

        log_weight_inc = (
            self.transition_log_prob(
                transition_history,
                proposed_particles,
                condition,
                target_parameters(params),
            )
            + self.emission_log_prob(
                emission_particles,
                observation,
                obs_history,
                condition,
                target_parameters(params),
            )
            - self.proposal_log_prob(
                proposal_history, observation, proposed_particles, condition, params
            )
        )

        particles = (*resampled_particles, proposed_particles)[
            -max(self.target.transition.order, self.target.emission.order) :
        ]

        return FilterData(
            start_log_w=log_w,
            resampled_log_w=resampled_log_w,
            log_w=resampled_log_w + log_weight_inc,
            particles=particles,
            ancestor_ix=ancestor_ix,
            log_w_inc=log_weight_inc,
        )


def run_filter[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](
    key: PRNGKeyArray,
    smc: SMCSampler[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
    ],
    parameters: ParametersT,
    observation_path: ObservationT,
    *,
    condition_path: ConditionT = seqjtyping.NoCondition(),
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

    initial_conditions = slice_prior_conditions(condition_path, smc.target.prior)
    observation_history = slice_emission_observation_history(
        observation_path,
        smc.target.emission,
    )

    init_key, *step_keys = jrandom.split(key, sequence_length)

    # Run initial step, this needs special handling because we sample from prior
    # rather than the proposal.
    init_particles = jax.vmap(smc.target.prior.sample, in_axes=[0, None, None])(
        jrandom.split(init_key, smc.num_particles),
        initial_conditions,
        target_parameters(parameters),
    )
    log_weights = smc.emission_log_prob(
        init_particles,
        util.index_pytree(observation_path, 0),
        observation_history,
        initial_conditions,
        target_parameters(parameters),
    )

    filter_data = FilterData(
        start_log_w=log_weights,
        resampled_log_w=log_weights,
        log_w=log_weights,
        particles=init_particles,
        ancestor_ix=jnp.full((smc.num_particles,), -1, dtype=jnp.int32),
        log_w_inc=-jnp.ones_like(log_weights) * jnp.log(smc.num_particles),
    )
    intial_record = (
        tuple(r(filter_data) for r in recorders) if recorders is not None else ()
    )

    # Define the main body
    def body(state, inputs):
        obs_hist = ()  # TODO slice_emission_observation_history(
        if sliced_condition_path is None:
            step_key, observation = inputs
            condition = None
        else:
            step_key, observation, condition = inputs
        log_w, particles = state
        step_data = smc.sample_step(
            step_key,
            log_w,
            particles,
            obs_hist,
            observation,
            condition,
            parameters,
            target_parameters,
        )

        recorder_vals = (
            tuple(r(step_data) for r in recorders) if recorders is not None else ()
        )

        return (step_data.log_w, step_data.particles), recorder_vals

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
        (filter_data.log_w, filter_data.particles),
        body_inputs,
    )

    def expand_concat(value, array):
        return jnp.concatenate([jnp.expand_dims(value, axis=0), array], axis=0)

    recorder_history = jax.tree_util.tree_map(
        expand_concat, intial_record, recorder_history
    )
    (log_w, particles) = final_state
    return (
        log_w,
        particles,
        recorder_history,
    )
