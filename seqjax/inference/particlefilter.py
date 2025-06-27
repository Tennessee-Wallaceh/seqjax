from typing import NamedTuple, TypeVar, Protocol, Generic, Optional, Callable
from jaxtyping import Array, Float, PRNGKeyArray
from seqjax.model.base import Target, ParticleType, ObservationType, ConditionType, ParametersType
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import Tuple
from jaxtyping import PRNGKeyArray, Scalar
from abc import abstractmethod

def sample_from_log_weights(key, log_weights, particles):
    # gumbel max trick
    gumbels = -jnp.log(-jnp.log(
        jrandom.uniform(key, (log_weights.shape[0], log_weights.shape[0]))
    ))
    particle_ix = jnp.argmax(log_weights + gumbels, axis=1).reshape(-1)
    return jax.vmap(
        index_tree, in_axes=[0, None],
    )(particle_ix, particles)

def compute_esse_from_log_weights(log_weights):
    # ess efficiency, ie ess / M
    log_w = log_weights - jnp.max(log_weights)  # for numerical stability
    log_sum_w = jax.scipy.special.logsumexp(log_weights)
    log_sum_w2 = jax.scipy.special.logsumexp(2 * log_weights)
    ess = jnp.exp(2 * log_sum_w - log_sum_w2)
    return ess / log_w.shape[0]

class Proposal(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType]
):
    order: eqx.AbstractClassVar[int]  # how many particles are output

    @staticmethod
    @abstractmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: Tuple[ParticleType, ...],
        observation_history: Tuple[ObservationType, ...],
        current_observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Tuple[ParticleType, ...]:
        ...

    @staticmethod
    @abstractmethod
    def log_p(
        particle_history: Tuple[ParticleType, ...],
        particle: Tuple[ParticleType, ...],             
        observation_history: Tuple[ObservationType, ...],
        current_observation: ObservationType,
        condition: ConditionType,
        parameters: ParametersType,
    ) -> Scalar:
        ...

def bootstrap_proposal(
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType]
) -> Proposal[ParticleType, ObservationType, ConditionType, ParametersType]:
    proposal_order = max(target.emission.order, target.transition.order)
    class BootstrapProposal(Proposal):
        order = target.transition.order

        @staticmethod
        def sample(
            key: PRNGKeyArray,
            particle_history: Tuple[ParticleType, ...],
            observation_history: Tuple[ObservationType, ...],
            current_observation: ObservationType,
            condition: ConditionType,
            parameters: ParametersType,
        ) -> Tuple[ParticleType, ...]:
            xt = target.transition.sample(
                key, particle_history, condition, parameters
            )
            print(proposal_order)
            return (*particle_history[-proposal_order:], xt)

        @staticmethod
        def log_p(
            particle_history: Tuple[ParticleType, ...],
            particle: Tuple[ParticleType, ...],
            observation_history: Tuple[ObservationType, ...],
            current_observation: ObservationType,
            condition: ConditionType,
            parameters: ParametersType,
        ) -> Scalar:
            return target.transition.log_p(
                particle_history, particle[-1], condition, parameters
            )

    return BootstrapProposal()

class GeneralSequentialImportanceSampler(
    eqx.Module,
    Generic[ParticleType, ObservationType, ConditionType, ParametersType]
):
    """
    Generic class encapsulating a gSIS setup.
    """
    target: Target[ParticleType, ObservationType, ConditionType, ParametersType]
    proposal: Proposal[ParticleType, ObservationType, ConditionType, ParametersType]
    # particles, log_weights to resampled indices
    resample: Callable[[jax.Array], jax.Array]  

def init_particles(
    sis: GeneralSequentialImportanceSampler[ParticleType, ObservationType, ConditionType, ParametersType],
    key: PRNGKeyArray,
    prior_conditions: Tuple[ConditionType, ...],
    parameters: ParametersType,
    num_particles: int,
) -> Tuple[Tuple[ParticleType, ...], Array]: 
    keys = jax.random.split(key, num_particles)
    
    vmapped_sample = jax.vmap(sis.target.prior.sample, in_axes=[0, None, None])
    vmapped_log_p = jax.vmap(sis.target.prior.log_p, in_axes=[0, None, None])
    particles = vmapped_sample(keys, prior_conditions, parameters)
    log_ps = vmapped_log_p(particles, prior_conditions, parameters)

    return particles, log_ps
   
def filter_step(
    sis: GeneralSequentialImportanceSampler[ParticleType, ObservationType, ConditionType, ParametersType],
    key: PRNGKeyArray,
    particle_history: Tuple[ParticleType, ...],  # Tuple of batched particles
    observation_history: Tuple[ObservationType, ...],
    current_observation: ObservationType,
    condition: ConditionType,
    parameters: ParametersType,
    num_particles: int,
) -> Tuple[Tuple[ParticleType, ...], Array]:  # (new particle history, new log weights)
    keys = jax.random.split(key, num_particles)

    sample_fn = lambda k, ph: sis.proposal.sample(
        k, 
        ph[-sis.target.transition.order:],
        observation_history,
        current_observation,
        condition,
        parameters,
    )
    new_particles = jax.vmap(sample_fn)(keys, particle_history) 

    def log_weight_fn(ph, np):
        transition_ph = ph[-sis.target.transition.order:]
        log_p_emit = sis.target.emission.log_p(
            np, observation_history, current_observation, condition, parameters
        )
        log_p_trans = sis.target.transition.log_p(
            transition_ph, np[-1], condition, parameters
        )
        log_q = sis.proposal.log_p(
            transition_ph, 
            np, 
            observation_history, 
            current_observation, 
            condition, 
            parameters
        )
        return log_p_emit + log_p_trans - log_q

    log_weights = jax.vmap(log_weight_fn)(particle_history, new_particles)

    return new_particles, log_weights

