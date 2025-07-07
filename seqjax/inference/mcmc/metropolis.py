from __future__ import annotations

from typing import Callable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray
import blackjax  # type: ignore

Parameters = TypeVar("Parameters")


class RandomWalkConfig(eqx.Module):
    """Configuration for :func:`run_random_walk_metropolis`."""

    step_size: float = 0.1
    num_samples: int = 100


LogDensity = Callable[[Parameters, PRNGKeyArray], jnp.ndarray]


def run_random_walk_metropolis(
    logdensity: LogDensity,
    key: PRNGKeyArray,
    initial_parameters: Parameters,
    config: RandomWalkConfig = RandomWalkConfig(),
) -> jax.Array | Parameters:
    """Run Random Walk Metropolis sampling using ``blackjax`` utilities."""

    init_key, *step_keys = jrandom.split(key, config.num_samples + 1)
    init_logp = logdensity(initial_parameters, init_key)

    random_step = blackjax.random_walk.normal(config.step_size)

    def step(state, rng):
        params, logp = state
        prop_key, ld_key, accept_key = jrandom.split(rng, 3)
        move = random_step(prop_key, params)
        proposal = jax.tree_util.tree_map(jnp.add, params, move)
        prop_logp = logdensity(proposal, ld_key)
        log_ratio = prop_logp - logp
        new_params, info = blackjax.mcmc.proposal.static_binomial_sampling(
            accept_key, log_ratio, params, proposal
        )
        do_accept, _, _ = info
        new_logp = jnp.where(do_accept, prop_logp, logp)
        return (new_params, new_logp), new_params

    _, samples = jax.lax.scan(step, (initial_parameters, init_logp), jnp.array(step_keys))
    return samples
