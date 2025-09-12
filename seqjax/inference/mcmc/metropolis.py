import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jaxtyping
import blackjax  # type: ignore
from jax_tqdm import scan_tqdm
import seqjax.model.typing as seqjtyping


class RandomWalkConfig(eqx.Module):
    """Configuration for :func:`run_random_walk_metropolis`."""

    step_size: float = 0.1


LogDensity = typing.Callable[
    [seqjtyping.Parameters, jaxtyping.PRNGKeyArray], jnp.ndarray
]


def run_random_walk_metropolis(
    logdensity: LogDensity,
    key: jaxtyping.PRNGKeyArray,
    initial_parameters: seqjtyping.Parameters,
    config: RandomWalkConfig = RandomWalkConfig(),
    num_samples: int = 1000,
) -> jax.Array | seqjtyping.Parameters:
    """Run Random Walk Metropolis sampling using ``blackjax`` utilities."""

    init_key, *step_keys = jrandom.split(key, num_samples + 1)
    init_logp = logdensity(initial_parameters, init_key)

    random_step = blackjax.random_walk.normal(jnp.array(config.step_size))

    @scan_tqdm(num_samples)
    def step(state, inputs):
        ix, params, logp = state
        _, rng = inputs
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
        return (ix + 1, new_params, new_logp), new_params

    step = typing.cast(
        typing.Callable[
            [
                tuple[int, seqjtyping.Parameters, jaxtyping.Array],
                tuple[jaxtyping.Array, jaxtyping.PRNGKeyArray],
            ],
            tuple[
                tuple[int, seqjtyping.Parameters, jaxtyping.Array],
                seqjtyping.Parameters,
            ],
        ],
        step,
    )

    _, samples = jax.lax.scan(
        step,
        (0, initial_parameters, init_logp),
        (jnp.arange(num_samples), jnp.array(step_keys)),
    )
    return samples
