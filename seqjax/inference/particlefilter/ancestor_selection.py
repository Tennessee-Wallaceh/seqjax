import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.special as jsp
from jaxtyping import Array, PRNGKeyArray, PyTree

from seqjax.model import interface as model_interface
import seqjax.model.typing as seqjtyping
from seqjax.model import util as model_util
from seqjax.util import index_pytree
from . import interface as pf_interface

class ResamplingAncestorSelection[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    FilterLatentContextLength: int,
    FilterObservationContextLength: int,
](eqx.Module):
    """Select ancestors using the current filtering weights."""

    resampler: pf_interface.Resampler
    latent_context_length: int = eqx.field(static=True, default=0)
    observation_context_length: int = eqx.field(static=True, default=0)

    def __call__(
        self,
        key: PRNGKeyArray,
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
        parameters: ParametersT,
        condition: ConditionT,
        num_particles: int,
    ) -> pf_interface.AncestorSelectionResult[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentContextLength,
        FilterObservationContextLength,
    ]:
        ancestor_sample = self.resampler(
            key, incoming.normalized_log_weights, num_particles
        )
        selected_particles = (
            incoming.context.particles
            if len(incoming.context.particles) == 0
            else jax.tree_util.tree_map(
                lambda x: jnp.take(x, ancestor_sample.ancestor_ix),
                incoming.context.particles,
            )
        )

        return pf_interface.AncestorSelectionResult(
            selected=pf_interface.WeightedPopulation(
                context=incoming.context.with_particles(selected_particles),
                normalized_log_weights=ancestor_sample.normalized_log_weights,
            ),
            ancestor_ix=ancestor_sample.ancestor_ix,
            selection_log_z_adjustment=jnp.zeros(
                (), dtype=incoming.normalized_log_weights.dtype
            ),
        )
    
class LookaheadAncestorSelection[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    ModelLatentContextLength: int,
    ModelObservationContextLength: int,
    FilterLatentContextLength: int,
    FilterObservationContextLength: int,
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
    resampler: pf_interface.Resampler
    latent_context_length: int = eqx.field(static=True)
    observation_context_length: int = eqx.field(static=True)

    def __call__(
        self,
        key: PRNGKeyArray,
        incoming: pf_interface.WeightedPopulation[
            pf_interface.FilterContext[
                ParticleT,
                ObservationT,
                ConditionT,
                FilterLatentContextLength,
                FilterObservationContextLength
            ]   
        ],
        observation: ObservationT,
        parameters: ParametersT,
        condition: ConditionT,
        num_particles: int,
    ) -> pf_interface.AncestorSelectionResult[
        ParticleT, ObservationT, ConditionT,
        FilterLatentContextLength, FilterObservationContextLength,
    ]:
        
        if incoming.context.length == 0:
            raise ValueError(
                "Emission lookahead requires a retained previous latent state"
            )

        latent_history = incoming.context.latent_context(
            self.target.latent_context_length
        )
        observation_history = incoming.context.observed_context(
            self.target.observation_context_length
        )
        current_particles = incoming.context.particles[-1]

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

        ancestor_log_weight = (
            incoming.normalized_log_weights + lookahead_log_weight
        )
        log_first_stage_norm = jsp.logsumexp(ancestor_log_weight)

        ancestor_sample = self.resampler(
            key,
            ancestor_log_weight - log_first_stage_norm,
            num_particles,
        )

        # Select the *full filter histories* using ancestor_sample.ancestor_ix.
        selected_context = incoming.context.with_particles(
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, ancestor_sample.ancestor_ix),
                incoming.context.particles,
            )
        )

        corrected_log_weight = (
            ancestor_sample.normalized_log_weights
            + log_first_stage_norm
            - lookahead_log_weight[ancestor_sample.ancestor_ix]
        )
        selection_log_z_adjustment = jsp.logsumexp(corrected_log_weight)

        return pf_interface.AncestorSelectionResult(
            selected=pf_interface.WeightedPopulation(
                context=selected_context,
                normalized_log_weights=(
                    corrected_log_weight - selection_log_z_adjustment
                ),
            ),
            ancestor_ix=ancestor_sample.ancestor_ix,
            selection_log_z_adjustment=selection_log_z_adjustment,
        )