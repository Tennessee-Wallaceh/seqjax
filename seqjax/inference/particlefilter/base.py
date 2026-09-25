import typing
from functools import cached_property

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jsp
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from seqjax.model import interface as model_interface
import seqjax.model.typing as seqjtyping
from seqjax.model import util as model_util
from .resampling import Resampler
from . import interface as pf_interface


class Proposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    ProposalLatentContextLength: int,
    ProposalObservationContextLength: int,
](typing.Protocol):
    """Full-population proposal kernel for sequential Monte Carlo."""

    latent_context_length: ProposalLatentContextLength
    observation_context_length: ProposalObservationContextLength

    def propose[
        FilterLatentHistoryLength: int,
        FilterObservationHistoryLength: int,
    ](
        self,
        key: PRNGKeyArray,
        start_log_w: Array,
        context: pf_interface.FilterContext[
            ParticleT,
            ObservationT,
            ConditionT,
            FilterLatentHistoryLength,
            FilterObservationHistoryLength,
        ],
        observation: ObservationT,
        parameters: ParametersT,
        condition: ConditionT,
        num_particles: int,
    ) -> pf_interface.ProposalResult[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ]: ...


class TransitionProposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    ModelLatentContextLength: int,
    ModelObservationContextLength: int,
](eqx.Module):
    """Use the model transition as a bootstrap-filter proposal kernel."""

    transition_sample_fn: typing.Callable[..., ParticleT]
    transition_log_prob_fn: typing.Callable[..., Scalar]
    resampler: Resampler
    latent_context_length: ModelLatentContextLength = eqx.field(static=True)
    observation_context_length: ModelObservationContextLength = eqx.field(static=True)

    def __init__(self, model, resampler: Resampler):
        self.transition_sample_fn = model.transition_sample
        self.transition_log_prob_fn = model.transition_log_prob
        self.resampler = resampler
        self.latent_context_length = typing.cast(
            ModelLatentContextLength, model.latent_context_length
        )
        self.observation_context_length = typing.cast(
            ModelObservationContextLength, model.observation_context_length
        )

    def _sample(
        self,
        key,
        latent_history,
        observation,
        parameters,
        condition,
        observation_history,
    ):
        del observation
        return self.transition_sample_fn(
            key,
            latent_history,
            parameters,
            condition,
            observation_history,
        )

    def _log_prob(
        self,
        latent_history,
        observation,
        particle,
        parameters,
        condition,
        observation_history,
    ):
        del observation
        return self.transition_log_prob_fn(
            particle,
            latent_history,
            parameters,
            condition,
            observation_history,
        )

    def propose(
        self,
        key,
        start_log_w,
        context,
        observation,
        parameters,
        condition,
        num_particles,
    ):
        resample_key, proposal_key = jrandom.split(key)
        ancestor_log_prob = start_log_w - jsp.logsumexp(start_log_w)
        resampled_history, ancestor_ix, resampled_log_w = self.resampler(
            resample_key, ancestor_log_prob, context.particles, num_particles
        )
        resampled_context = context.with_particles(resampled_history)
        latent_history = resampled_context.latent_context(self.latent_context_length)
        observation_history = resampled_context.observed_context(
            self.observation_context_length
        )
        particles = jax.vmap(self._sample, in_axes=(0, 0, None, None, None, None))(
            jrandom.split(proposal_key, num_particles),
            latent_history,
            observation,
            parameters,
            condition,
            observation_history,
        )
        proposal_log_prob = jax.vmap(
            self._log_prob, in_axes=(0, None, 0, None, None, None)
        )(
            latent_history,
            observation,
            particles,
            parameters,
            condition,
            observation_history,
        )
        return pf_interface.ProposalResult(
            particles=particles,
            resampled_context=resampled_context,
            ancestor_ix=ancestor_ix,
            ancestor_log_prob=ancestor_log_prob,
            resampled_log_w=resampled_log_w,
            proposal_log_prob=proposal_log_prob,
        )


class AuxiliaryTransitionProposal(TransitionProposal):
    """Use one transition draw per ancestor for auxiliary look-ahead weights."""

    emission_log_prob_fn: typing.Callable[..., Scalar]

    def __init__(self, model, resampler: Resampler):
        self.emission_log_prob_fn = model.emission_log_prob
        super().__init__(model, resampler)

    def propose(
        self,
        key,
        start_log_w,
        context,
        observation,
        parameters,
        condition,
        num_particles,
    ):
        lookahead_key, resample_key, proposal_key = jrandom.split(key, 3)
        latent_history = context.latent_context(self.latent_context_length)
        observation_history = context.observed_context(self.observation_context_length)
        lookahead_particles = jax.vmap(
            self._sample, in_axes=(0, 0, None, None, None, None)
        )(
            jrandom.split(lookahead_key, num_particles),
            latent_history,
            observation,
            parameters,
            condition,
            observation_history,
        )
        auxiliary_log_weights = jax.vmap(
            self.emission_log_prob_fn,
            in_axes=(None, 0, None, None, 0, None),
        )(
            observation,
            lookahead_particles,
            parameters,
            condition,
            latent_history,
            observation_history,
        )
        first_stage_log_w = start_log_w + auxiliary_log_weights
        ancestor_log_prob = first_stage_log_w - jsp.logsumexp(first_stage_log_w)
        resampled_history, ancestor_ix, resampled_log_w = self.resampler(
            resample_key, ancestor_log_prob, context.particles, num_particles
        )
        resampled_context = context.with_particles(resampled_history)
        selected_latent_history = resampled_context.latent_context(
            self.latent_context_length
        )
        particles = jax.vmap(self._sample, in_axes=(0, 0, None, None, None, None))(
            jrandom.split(proposal_key, num_particles),
            selected_latent_history,
            observation,
            parameters,
            condition,
            observation_history,
        )
        proposal_log_prob = jax.vmap(
            self._log_prob, in_axes=(0, None, 0, None, None, None)
        )(
            selected_latent_history,
            observation,
            particles,
            parameters,
            condition,
            observation_history,
        )
        return pf_interface.ProposalResult(
            particles=particles,
            resampled_context=resampled_context,
            ancestor_ix=ancestor_ix,
            ancestor_log_prob=ancestor_log_prob,
            resampled_log_w=resampled_log_w,
            proposal_log_prob=proposal_log_prob,
        )


class SMCSampler[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    ModelLatentContextLength: int = int,
    ModelObservationContextLength: int = int,
    ProposalLatentContextLength: int = int,
    ProposalObservationContextLength: int = int,
    FilterLatentHistoryLength: int = int,
    FilterObservationHistoryLength: int = int,
](eqx.Module):
    """Base sequential Monte Carlo sampler with algorithm-level histories."""

    target: model_interface.SequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
        ModelLatentContextLength,
        ModelObservationContextLength,
    ]
    proposal: Proposal[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
        ProposalLatentContextLength,
        ProposalObservationContextLength,
    ]
    num_particles: int = eqx.field(static=True)
    latent_context_length: FilterLatentHistoryLength = eqx.field(static=True)
    observation_context_length: FilterObservationHistoryLength = eqx.field(static=True)

    def __init__(
        self,
        *,
        target: model_interface.SequentialModelProtocol[
            ParticleT,
            ObservationT,
            ConditionT,
            ParameterT,
            ModelLatentContextLength,
            ModelObservationContextLength,
        ],
        proposal: Proposal[
            ParticleT,
            ObservationT,
            ConditionT,
            ParameterT,
            ProposalLatentContextLength,
            ProposalObservationContextLength,
        ],
        num_particles: int,
        latent_context_length: FilterLatentHistoryLength | None = None,
        observation_context_length: FilterObservationHistoryLength | None = None,
    ):
        self.target = target
        self.proposal = proposal
        self.num_particles = num_particles
        self.latent_context_length = typing.cast(
            FilterLatentHistoryLength,
            target.latent_context_length
            if latent_context_length is None
            else latent_context_length,
        )
        self.observation_context_length = typing.cast(
            FilterObservationHistoryLength,
            target.observation_context_length
            if observation_context_length is None
            else observation_context_length,
        )
        self._validate_context_lengths()

    def _validate_context_lengths(self) -> None:
        requirements = (
            (
                "latent",
                self.latent_context_length,
                max(
                    self.target.latent_context_length,
                    self.proposal.latent_context_length,
                ),
            ),
            (
                "observation",
                self.observation_context_length,
                max(
                    self.target.observation_context_length,
                    self.proposal.observation_context_length,
                ),
            ),
        )
        for name, actual, required in requirements:
            if actual < required:
                raise ValueError(
                    f"Filter {name} context length must be at least {required}; "
                    f"received {actual}"
                )

    @cached_property
    def transition_log_prob(self) -> typing.Callable[..., Array]:
        return jax.vmap(
            self.target.transition_log_prob, in_axes=(0, 0, None, None, None)
        )

    @cached_property
    def emission_log_prob(self) -> typing.Callable[..., Array]:
        return jax.vmap(
            self.target.emission_log_prob,
            in_axes=(None, 0, None, None, 0, None),
        )

    def filter_context(
        self,
        particles: tuple[ParticleT, ...],
        observations: tuple[
            model_interface.ObservedItem[ObservationT, ConditionT],
            ...,
        ],
    ) -> pf_interface.FilterContext[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ]:
        return pf_interface.FilterContext(
            particles=model_interface.LatentContext.from_values(
                *particles,
                length=self.latent_context_length,
            ),
            observations=model_interface.ObservedHistoryContext.from_values(
                *observations,
                length=self.observation_context_length,
            ),
        )

    def sample_step(
        self,
        step_ix: int,
        step_key: PRNGKeyArray,
        start_log_w: Array,
        context: pf_interface.FilterContext[
            ParticleT,
            ObservationT,
            ConditionT,
            FilterLatentHistoryLength,
            FilterObservationHistoryLength,
        ],
        observation: ObservationT,
        params: ParameterT,
        condition: ConditionT,
    ) -> pf_interface.FilterData[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ]:
        proposal_result = self.proposal.propose(
            step_key,
            start_log_w,
            context,
            observation,
            params,
            condition,
            self.num_particles,
        )
        resampled_context = proposal_result.resampled_context
        proposed_particles = proposal_result.particles

        model_latent_history = resampled_context.latent_context(
            typing.cast(
                ModelLatentContextLength,
                self.target.latent_context_length,
            )
        )
        model_observation_history = resampled_context.observed_context(
            typing.cast(
                ModelObservationContextLength,
                self.target.observation_context_length,
            )
        )
        ancestor_log_weight = (
            start_log_w[proposal_result.ancestor_ix]
            - proposal_result.ancestor_log_prob[proposal_result.ancestor_ix]
        )
        log_weight_inc = (
            self.transition_log_prob(
                proposed_particles,
                model_latent_history,
                params,
                condition,
                model_observation_history,
            )
            + self.emission_log_prob(
                observation,
                proposed_particles,
                params,
                condition,
                model_latent_history,
                model_observation_history,
            )
            - proposal_result.proposal_log_prob
            + ancestor_log_weight
        )
        next_context = resampled_context.append(
            proposed_particles,
            observation,
            condition,
        )

        log_w_unnorm = proposal_result.resampled_log_w + log_weight_inc
        log_z_inc = jsp.logsumexp(log_w_unnorm)
        log_w = log_w_unnorm - log_z_inc

        return pf_interface.FilterData(
            step_ix=step_ix,
            start_log_w=start_log_w,
            resampled_log_w=proposal_result.resampled_log_w,
            log_w=log_w,
            particles=next_context,
            ancestor_ix=proposal_result.ancestor_ix,
            log_w_inc=log_weight_inc,
            resampled_particles=resampled_context,
            observation=observation,
            condition=condition,
            inference_parameters=params,
            log_z_inc=log_z_inc,
        )


def run_filter[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    ModelLatentContextLength: int,
    ModelObservationContextLength: int,
    ProposalLatentContextLength: int,
    ProposalObservationContextLength: int,
    FilterLatentHistoryLength: int,
    FilterObservationHistoryLength: int,
](
    key: PRNGKeyArray,
    smc: SMCSampler[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
        ModelLatentContextLength,
        ModelObservationContextLength,
        ProposalLatentContextLength,
        ProposalObservationContextLength,
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ],
    parameters: ParameterT,
    observation_path: ObservationT,
    condition_path: ConditionT | None = None,
    observation_history: model_interface.FixedLengthHistoryContext[
        model_interface.ObservedItem[ObservationT, ConditionT],
        FilterObservationHistoryLength,
    ]
    | None = None,
    *,
    recorders: tuple[pf_interface.Recorder, ...] | None = None,
) -> tuple[
    Array,
    pf_interface.FilterContext[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ],
    tuple[PyTree, ...],
]:
    """Run a filtering pass over ``observation_path``."""

    sequence_length = observation_path.batch_shape[0]
    condition_path = model_util.normalize_condition_path(
        smc.target,
        condition_path,
        (sequence_length,),
    )
    init_key, *step_keys = jrandom.split(key, sequence_length + 1)

    prior_particles = jax.vmap(
        smc.target.prior_sample,
        in_axes=(0, None),
    )(
        jrandom.split(init_key, smc.num_particles),
        parameters,
    )
    if prior_particles.length != smc.latent_context_length:
        raise ValueError(
            "The model prior supplies latent context length "
            f"{prior_particles.length}, but the filter requires "
            f"{smc.latent_context_length}"
        )

    if observation_history is None:
        if smc.observation_context_length != 0:
            raise ValueError(
                "observation_history is required because the filter "
                f"observation context length is {smc.observation_context_length}"
            )
        observation_history = smc.target.observed_history_context()
    else:
        if observation_history.length != smc.observation_context_length:
            raise ValueError(
                "observation_history has the wrong length: expected "
                f"{smc.observation_context_length}, received "
                f"{observation_history.length}"
            )
    context = smc.filter_context(prior_particles.values, observation_history.values)

    uniform_log_w = jnp.full(
        (smc.num_particles,),
        -jnp.log(smc.num_particles),
    )

    def body(state, inputs):
        log_w, filter_context = state
        step_ix, step_key, observation, condition = inputs
        step_data = smc.sample_step(
            step_ix,
            step_key,
            log_w,
            filter_context,
            observation,
            parameters,
            condition,
        )
        recorder_values = (
            tuple(recorder(step_data) for recorder in recorders)
            if recorders is not None
            else ()
        )
        return (step_data.log_w, step_data.particles), recorder_values

    body_inputs = (
        jnp.arange(sequence_length),
        jnp.array(step_keys),
        observation_path,
        condition_path,
    )
    (log_w, context), recorder_history = jax.lax.scan(
        body,
        init=(uniform_log_w, context),
        xs=body_inputs,
    )
    return log_w, context, recorder_history
