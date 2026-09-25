import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jsp
from jaxtyping import Array, PRNGKeyArray, PyTree

from seqjax.model import interface as model_interface
import seqjax.model.typing as seqjtyping
from seqjax.model import util as model_util
from seqjax.util import dynamic_index_pytree_in_dim as index_tree
from .resampling import Resampler
from . import interface as pf_interface


class Proposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    FilterLatentHistoryLength: int,
    FilterObservationHistoryLength: int,
](typing.Protocol):
    """Population mutation kernel for one complete SMC step."""

    latent_context_length: int
    observation_context_length: int

    def __call__(
        self,
        key: PRNGKeyArray,
        start_log_weight: Array,
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


def _resample_context(
    resampler,
    key,
    ancestor_log_prob,
    context,
    num_particles,
):
    ancestor_sample = resampler(key, ancestor_log_prob, num_particles)
    resampled_particles = (
        context.particles
        if len(context.particles) == 0
        else jax.vmap(index_tree, in_axes=(None, 0, None))(
            context.particles,
            ancestor_sample.indices,
            0,
        )
    )
    return context.with_particles(resampled_particles), ancestor_sample


def _transition_and_emission_log_prob(
    target,
    key,
    resampled_context,
    observation,
    parameters,
    condition,
    num_particles,
):
    latent_history = resampled_context.latent_context(target.latent_context_length)
    observation_history = resampled_context.observed_context(
        target.observation_context_length
    )
    proposed_particles = jax.vmap(
        target.transition_sample,
        in_axes=(0, 0, None, None, None),
    )(
        jrandom.split(key, num_particles),
        latent_history,
        parameters,
        condition,
        observation_history,
    )
    emission_log_prob = jax.vmap(
        target.emission_log_prob,
        in_axes=(None, 0, None, None, 0, None),
    )(
        observation,
        proposed_particles,
        parameters,
        condition,
        latent_history,
        observation_history,
    )
    return proposed_particles, emission_log_prob


def _proposal_result(
    resampled_context,
    proposed_particles,
    observation,
    condition,
    ancestor_sample,
    second_stage_log_weight,
    first_stage_log_normalizer,
):
    unnormalized_log_weight = ancestor_sample.log_weights + second_stage_log_weight
    second_stage_log_normalizer = jsp.logsumexp(unnormalized_log_weight)
    return pf_interface.ProposalResult(
        particles=resampled_context.append(
            proposed_particles,
            observation,
            condition,
        ),
        resampled_history=resampled_context,
        ancestor_indices=ancestor_sample.indices,
        log_weight=unnormalized_log_weight - second_stage_log_normalizer,
        log_normalizer_increment=(
            first_stage_log_normalizer + second_stage_log_normalizer
        ),
    )


class TransitionProposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    ModelLatentContextLength: int,
    ModelObservationContextLength: int,
](eqx.Module):
    """Bootstrap population kernel using the model transition proposal."""

    target: model_interface.SequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        ModelLatentContextLength,
        ModelObservationContextLength,
    ]
    resampler: Resampler
    latent_context_length: int = eqx.field(static=True)
    observation_context_length: int = eqx.field(static=True)

    def __call__(
        self,
        key,
        start_log_weight,
        context,
        observation,
        parameters,
        condition,
        num_particles,
    ):
        resample_key, proposal_key = jrandom.split(key)
        first_stage_log_normalizer = jsp.logsumexp(start_log_weight)
        ancestor_log_prob = start_log_weight - first_stage_log_normalizer
        resampled_context, ancestor_sample = _resample_context(
            self.resampler,
            resample_key,
            ancestor_log_prob,
            context,
            num_particles,
        )
        proposed_particles, emission_log_prob = _transition_and_emission_log_prob(
            self.target,
            proposal_key,
            resampled_context,
            observation,
            parameters,
            condition,
            num_particles,
        )
        return _proposal_result(
            resampled_context,
            proposed_particles,
            observation,
            condition,
            ancestor_sample,
            emission_log_prob,
            first_stage_log_normalizer,
        )


class AuxiliaryTransitionProposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    ModelLatentContextLength: int,
    ModelObservationContextLength: int,
](eqx.Module):
    """Transition kernel with emission-based auxiliary ancestor selection."""

    target: model_interface.SequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        ModelLatentContextLength,
        ModelObservationContextLength,
    ]
    resampler: Resampler
    latent_context_length: int = eqx.field(static=True)
    observation_context_length: int = eqx.field(static=True)

    def __call__(
        self,
        key,
        start_log_weight,
        context,
        observation,
        parameters,
        condition,
        num_particles,
    ):
        if context.length == 0:
            raise ValueError("Auxiliary proposals require at least one latent value")

        resample_key, proposal_key = jrandom.split(key)
        latent_history = context.latent_context(self.target.latent_context_length)
        observation_history = context.observed_context(
            self.target.observation_context_length
        )
        current_particles = context.particles[-1]
        lookahead_log_weight = jax.vmap(
            self.target.emission_log_prob,
            in_axes=(None, 0, None, None, 0, None),
        )(
            observation,
            current_particles,
            parameters,
            condition,
            latent_history,
            observation_history,
        )
        ancestor_logits = start_log_weight + lookahead_log_weight
        first_stage_log_normalizer = jsp.logsumexp(ancestor_logits)
        ancestor_log_prob = ancestor_logits - first_stage_log_normalizer
        resampled_context, ancestor_sample = _resample_context(
            self.resampler,
            resample_key,
            ancestor_log_prob,
            context,
            num_particles,
        )
        proposed_particles, emission_log_prob = _transition_and_emission_log_prob(
            self.target,
            proposal_key,
            resampled_context,
            observation,
            parameters,
            condition,
            num_particles,
        )
        second_stage_log_weight = (
            emission_log_prob - lookahead_log_weight[ancestor_sample.indices]
        )
        return _proposal_result(
            resampled_context,
            proposed_particles,
            observation,
            condition,
            ancestor_sample,
            second_stage_log_weight,
            first_stage_log_normalizer,
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
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
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
            FilterLatentHistoryLength,
            FilterObservationHistoryLength,
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
        proposal_result = self.proposal(
            step_key,
            start_log_w,
            context,
            observation,
            params,
            condition,
            self.num_particles,
        )

        return pf_interface.FilterData(
            step_ix=step_ix,
            start_log_w=start_log_w,
            log_w=proposal_result.log_weight,
            particles=proposal_result.particles,
            ancestor_ix=proposal_result.ancestor_indices,
            resampled_particles=proposal_result.resampled_history,
            observation=observation,
            condition=condition,
            inference_parameters=params,
            log_z_inc=proposal_result.log_normalizer_increment,
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
