from typing import NamedTuple, TypeVar, Protocol, Generic, Optional, Callable
from jaxtyping import Array, Float, PRNGKeyArray
from seqjax.target.base import Target, Particle, Observation, Condition, Parameters]
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrandom

def resample_from_log_weights(key, log_weights, particles):
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

class SampleStep(Protocol):
    def sample_step(
        key, 
        log_weights, 
        particles, 
        observation, condition, parameters
    ) -> log_weights, next_particles, ess_e:
        
class GeneralSequentialImportanceSampler(Protocol, Generic[Particle, Observation, Condition, Parameters]): 
    # The idea is that the SIS algorithm should be a function of the target and a 
    # proposal.
    
    @staticmethod
    def configure_filter(
        target: Target[Particle, Observation, Condition, Parameters], 
        num_particles: int, 
        key: PRNGKeyArray
    ) -> : ...