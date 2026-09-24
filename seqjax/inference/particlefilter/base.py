import typing
from functools import cached_property, partial
from typing import Callable
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jsp
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
from seqjax.model import interface as model_interface

import seqjax.model.typing as seqjtyping
from seqjax.model import util as model_util
from seqjax.model.condition import layout_for, normalize_condition_path
from seqjax import util
from .resampling import Resampler
from . import interface as pf_interface

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
    proposal_context: Callable[
        [tuple[ParticleT, ...]], 
        pf_interface.ProposalContext[ParticleT]
    ]

    @abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        particle_history: pf_interface.ProposalContext[ParticleT],
        observation: ObservationT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> ParticleT: ...

    """
    Leading axis is num particles
    """

    @abstractmethod
    def log_prob(
        self,
        particle_history: pf_interface.ProposalContext[ParticleT],
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
    TransitionLatentHistoryT: tuple[seqjtyping.Latent, ...],
](
    Proposal[ParticleT, ObservationT, ConditionT, ParametersT],
):
    """
    Wraps a SSM ``Transition`` to a ``Proposal``.
    This is what is done to produce the "Bootstrap" particle filter.
    We also supply an optional resampling scheme.
    If there is no resampling scheme this is SIS.
    """

    transition_sample_fn: Callable[..., ParticleT]
    transition_log_prob_fn: Callable[..., Scalar]
    # I don't actually care what this does, as long as it produces ParametersT.
    convert_to_model_parameters: Callable[[InferenceParametersT], ParametersT]
    proposal_context: Callable[
        [tuple[ParticleT, ...]],
        pf_interface.ProposalContext[ParticleT]
    ]

    def __init__(
        self,
        model: model_interface.BayesianSequentialModelProtocol[
            ParticleT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
        ],
    ):
        self.transition_sample_fn = model.target.transition_sample
        self.transition_log_prob_fn = model.target.transition_log_prob
        self.convert_to_model_parameters = model.parameterization.to_model_parameters
        super().__init__(order=model.target.transition_order, proposal_context=partial(pf_interface.ProposalContext, length=model.target.transition_order))

    def sample(
        self,
        key: PRNGKeyArray,
        particle_history: TransitionLatentHistoryT,
        observation: ObservationT,
        condition: ConditionT,
        parameters: InferenceParametersT,
    ) -> ParticleT:
        return self.transition_sample_fn(
            key, particle_history, condition, self.convert_to_model_parameters(parameters)
        )

    def log_prob(
        self,
        particle_history: TransitionLatentHistoryT,
        observation: ObservationT,
        new_particles: ParticleT,
        condition: ConditionT,
        parameters: InferenceParametersT,
    ) -> Array:
        return self.transition_log_prob_fn(
            particle_history,
            new_particles,
            condition,
            self.convert_to_model_parameters(parameters),
        )

class SMCSampler[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    InferenceParameterT: seqjtyping.Parameters
](
    eqx.Module,
):
    """Base class implementing sequential Monte Carlo."""
    target: model_interface.SequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
    ]
    parameterization: model_interface.ParameterizationProtocol[ParameterT, InferenceParameterT, typing.Any]
    proposal: Proposal[ParticleT, ObservationT, ConditionT, ParameterT]
    resampler: Resampler[ParticleT]
    num_particles: int

    @cached_property
    def proposal_sample(self) -> Callable:
        return jax.vmap(self.proposal.sample, in_axes=[0, 0, None, None, None])

    @cached_property
    def proposal_log_prob(self) -> Callable:
        return jax.vmap(self.proposal.log_prob, in_axes=[0, None, 0, None, None])

    @cached_property
    def transition_log_prob(self) -> Callable:
        return jax.vmap(self.target.transition_log_prob, in_axes=[0, 0, None, None])

    @cached_property
    def emission_log_prob(self) -> Callable:
        return jax.vmap(
            self.target.emission_log_prob,
            in_axes=[0, None, None, None, None],
        )

    def filter_context(self, particles: tuple[ParticleT, ...]) -> pf_interface.FilterContext[ParticleT]:
        return pf_interface.FilterContext(
            particles,
            length=max(
                self.target.transition_order,
                self.target.emission_order,
                self.proposal.order
            ) 
        )

    def sample_step(
        self,
        step_ix: int,
        step_key: PRNGKeyArray,
        start_log_w: Array,
        particles: pf_interface.FilterContext[ParticleT],
        observation: ObservationT,
        params: InferenceParameterT,
        condition: ConditionT,
        observation_history: model_interface.ObservedHistoryContext[
            ObservationT,
            ConditionT,
        ]
    ) -> pf_interface.FilterData:
        resample_key, proposal_key = jrandom.split(step_key)

        resampled_particles, ancestor_ix, resampled_log_w, _ = self.resampler(
            resample_key,
            start_log_w,
            particles,
            self.num_particles,
        )

        proposal_history = self.proposal.proposal_context(resampled_particles.values)
        transition_history = self.target.latent_context(resampled_particles.values)

        proposed_particles = self.proposal_sample(
            jrandom.split(proposal_key, self.num_particles),
            proposal_history,
            observation,
            condition,
            params,
        )

        emission_particles = transition_history.append(proposed_particles)

        obs_history = (
            observation_history[-self.target.observation_dependency :]
            if self.target.observation_dependency > 0
            else ()
        )

        model_params = self.parameterization.to_model_parameters(params)
        log_weight_inc = (
            self.transition_log_prob(
                transition_history,
                proposed_particles,
                condition,
                model_params,
            )
            + self.emission_log_prob(
                emission_particles,
                observation,
                obs_history,
                condition,
                model_params,
            )
            - self.proposal_log_prob(
                proposal_history, observation, proposed_particles, condition, params
            )
        )
        particles = resampled_particles.append(proposed_particles)

        log_w_unnorm = resampled_log_w + log_weight_inc
        log_z_inc = jsp.logsumexp(log_w_unnorm)
        log_w = log_w_unnorm - log_z_inc

        return pf_interface.FilterData(
            step_ix=step_ix,
            start_log_w=start_log_w,
            resampled_log_w=resampled_log_w,
            log_w=log_w,
            particles=particles,
            ancestor_ix=ancestor_ix,
            log_w_inc=log_weight_inc,
            resampled_particles=resampled_particles,
            observation=observation,
            condition=condition,
            inference_parameters=params,
            log_z_inc=log_z_inc
        )


def run_filter[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    InferenceParameterT: seqjtyping.Parameters,
](
    key: PRNGKeyArray,
    smc: SMCSampler[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
        InferenceParameterT,
    ],
    inference_parameters: InferenceParameterT,
    observation_path: ObservationT,
    condition_path: ConditionT | None = None,
    observation_history: model_interface.ObservedHistoryContext[
        ObservationT,
        ConditionT,
    ] | None = None,
    *,
    recorders: tuple[pf_interface.Recorder, ...] | None = None,
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

    sequence_length = observation_path.batch_shape[0]

    condition_path = normalize_condition_path(
        smc.target, condition_path, (sequence_length,)
    )

    init_key, *step_keys = jrandom.split(key, sequence_length)

    # Run initial step, this needs special handling because we sample from prior
    # rather than the proposal.
    model_parameters = smc.parameterization.to_model_parameters(
        inference_parameters
    )

    # Sample the latent context ending at t=-1.
    prior_particles = jax.vmap(
        smc.target.prior_sample,
        in_axes=(0, None),
    )(
        jrandom.split(init_key, smc.num_particles),
        model_parameters,
    )

    uniform_log_w = jnp.full(
        (smc.num_particles,),
        -jnp.log(smc.num_particles),
    )

    start_context = smc.filter_context(prior_particles.values)
    
    def body(
        state: tuple[
            Array,
            pf_interface.FilterContext[ParticleT],
        ],
        inputs: tuple[
            int,
            PRNGKeyArray,
            ObservationT,
            ConditionT,
            model_interface.ObservedHistoryContext[
                ObservationT,
                ConditionT,
            ]
        ],
    ):
        (
            step_ix,
            step_key,
            observation,
            condition,
            observation_history,
        ) = inputs

        log_w, particles = state

        step_data = smc.sample_step(
            step_ix,
            step_key,
            log_w,
            particles,
            observation,
            inference_parameters,
            condition,
            observation_history,
        )

        recorder_values = (
            tuple(recorder(step_data) for recorder in recorders)
            if recorders is not None
            else ()
        )

        return (
            step_data.log_w,
            step_data.particles,
        ), recorder_values

    batched_observation_history = model_util.batch_observation_history(
        smc.target,
        observation_history,
        observation_path,
        condition_path,
    )
    body_inputs = (
        jnp.arange(sequence_length),
        jnp.array(step_keys),
        observation_path,
        condition_path,
        batched_observation_history,
    )
    init_state = (uniform_log_w, start_context)

    (log_w, particles), recorder_history = jax.lax.scan(
        body,
        init=init_state,
        xs=body_inputs,
    )

    return (
        log_w,
        particles,
        recorder_history,
    )
