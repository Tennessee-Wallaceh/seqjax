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

class TransitionProposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    FilterLatentHistoryLength: int,
    FilterObservationHistoryLength: int,
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
    latent_context_length: int = eqx.field(static=True)
    observation_context_length: int = eqx.field(static=True)
    
    def __call__(
        self,
        key: PRNGKeyArray,
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
    ) -> pf_interface.ProposalResult[ParticleT] :
        latent_history = context.latent_context(
            self.target.latent_context_length
        )
        observation_history = context.observed_context(
            self.target.observation_context_length
        )

        particles = jax.vmap(
            self.target.transition_sample,
            in_axes=(0, 0, None, None, None),
        )(
            jrandom.split(key, num_particles),
            latent_history,
            parameters,
            condition,
            observation_history,
        )
        log_prob = jax.vmap(
            self.target.transition_log_prob,
            in_axes=(0, 0, None, None, None),
        )(
            particles,
            latent_history,
            parameters,
            condition,
            observation_history,
        )
        return pf_interface.ProposalResult(
            particles=particles,
            log_prob=log_prob,
        )