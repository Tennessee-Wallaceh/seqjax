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
from . import interface as pf_interface

class SMCSampler[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    ModelLatentContextLength: int = int,
    ModelObservationContextLength: int = int,
    FilterLatentContextLength: int = int,
    FilterObservationContextLength: int = int,
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
    proposal: pf_interface.Proposal[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
        FilterLatentContextLength,
        FilterObservationContextLength,
    ]
    ancestor_selection: pf_interface.AncestorSelection[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
        FilterLatentContextLength,
        FilterObservationContextLength,
    ]
    num_particles: int = eqx.field(static=True)

    @property
    def latent_context_length(self) -> FilterLatentContextLength:
        return typing.cast(
            FilterLatentContextLength,
            max(
                self.target.latent_context_length,
                self.proposal.latent_context_length,
                self.ancestor_selection.latent_context_length,
            ),
        )

    @property
    def observation_context_length(self) -> FilterObservationContextLength:
        return typing.cast(
            FilterObservationContextLength,
            max(
                self.target.observation_context_length,
                self.proposal.observation_context_length,
                self.ancestor_selection.observation_context_length,
            ),
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
        FilterLatentContextLength,
        FilterObservationContextLength,
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
        incoming: pf_interface.WeightedPopulation[
            pf_interface.FilterContext[
                ParticleT,
                ObservationT,
                ConditionT,
                FilterLatentContextLength,
                FilterObservationContextLength,
            ]
        ],
        observation: ObservationT,
        params: ParameterT,
        condition: ConditionT,
    ) -> pf_interface.FilterData[
        ParticleT,
        ObservationT,
        ConditionT,
        ParameterT,
        FilterLatentContextLength,
        FilterObservationContextLength,
    ]:
        selection_key, proposal_key = jrandom.split(step_key)

        selection_result = self.ancestor_selection(
            selection_key,
            incoming,
            observation,
            params,
            condition,
            self.num_particles,
        )
        selected = selection_result.selected

        proposal_result = self.proposal(
            proposal_key,
            selected.context,
            observation,
            params,
            condition,
            self.num_particles,
        )

        latent_history = selected.context.latent_context(
            self.target.latent_context_length
        )
        observation_history = selected.context.observed_context(
            self.target.observation_context_length
        )

        transition_log_prob = jax.vmap(
            self.target.transition_log_prob,
            in_axes=(0, 0, None, None, None),
        )(
            proposal_result.particles,
            latent_history,
            params,
            condition,
            observation_history,
        )
        emission_log_prob = jax.vmap(
            self.target.emission_log_prob,
            in_axes=(None, 0, None, None, 0, None),
        )(
            observation,
            proposal_result.particles,
            params,
            condition,
            latent_history,
            observation_history,
        )

        log_unnormalized_w = (
            selected.normalized_log_weights
            + transition_log_prob
            + emission_log_prob
            - proposal_result.log_prob
        )
        log_weight_norm = jsp.logsumexp(log_unnormalized_w)

        updated = pf_interface.WeightedPopulation(
            context=selected.context.append(
                proposal_result.particles, observation, condition
            ),
            normalized_log_weights=log_unnormalized_w - log_weight_norm,
        )

        return pf_interface.FilterData(
            step_ix=step_ix,
            incoming=incoming,
            selected=selected,
            updated=updated,
            ancestor_ix=selection_result.ancestor_ix,
            selection_log_z_adjustment=selection_result.selection_log_z_adjustment,
            log_z_inc=selection_result.selection_log_z_adjustment + log_weight_norm,
            observation=observation,
            condition=condition,
            parameters=params,
        )


def run_filter[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    ModelLatentContextLength: int,
    ModelObservationContextLength: int,
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
        step_ix, step_key, observation, condition = inputs
        new_state = smc.sample_step(
            step_ix,
            step_key,
            state,
            observation,
            parameters,
            condition,
        )
        recorder_values = (
            tuple(recorder(new_state) for recorder in recorders)
            if recorders is not None
            else ()
        )
        return new_state.updated, recorder_values

    body_inputs = (
        jnp.arange(sequence_length),
        jnp.array(step_keys),
        observation_path,
        condition_path,
    )
    final_state, recorder_history = jax.lax.scan(
        body,
        init=pf_interface.WeightedPopulation(
            context=context,
            normalized_log_weights=uniform_log_w,
        ),
        xs=body_inputs,
    )
    return (
        final_state.normalized_log_weights, 
        final_state.context, 
        recorder_history,
    )
