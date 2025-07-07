import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import simulate
from seqjax.model.stochastic_vol import (
    SkewStochasticVol,
    LogVolWithSkew,
    TimeIncrement,
)
import jax

from seqjax.inference.particlefilter import (
    BootstrapParticleFilter,
    vmapped_run_filter,
    log_marginal,
)


if __name__ == "__main__":
    steps = 200
    beta_true = -0.5
    params = LogVolWithSkew(
        std_log_vol=jnp.array(3.2),
        mean_reversion=jnp.array(12.0),
        long_term_vol=jnp.array(0.16),
        skew=jnp.array(beta_true),
    )

    dt = jnp.array(1.0 / (256 * 8))
    target = SkewStochasticVol()
    cond = TimeIncrement(dt * jnp.ones(steps + target.prior.order))
    key = jrandom.key(0)
    latent, obs, *_ = simulate.simulate(
        key,
        target,
        cond,
        params,
        sequence_length=steps,
    )

    betas = jnp.array([-0.8, -0.5, -0.2, 0.0])
    num_repeats = 50

    init_conds = tuple(TimeIncrement(cond.dt[i]) for i in range(target.prior.order))
    cond_path = TimeIncrement(cond.dt[target.prior.order:])
    bpf = BootstrapParticleFilter(target, num_particles=500)

    fig, axes = plt.subplots(1, betas.shape[0], figsize=(12, 3), sharey=True)
    base_key = jrandom.key(1)
    for beta, ax in zip(betas, axes):
        if hasattr(params, "replace"):
            par = params.replace(skew=jnp.array(beta))
        else:
            par = LogVolWithSkew(
                std_log_vol=params.std_log_vol,
                mean_reversion=params.mean_reversion,
                long_term_vol=params.long_term_vol,
                skew=jnp.array(beta),
            )

        split_keys = jrandom.split(base_key, num_repeats + 1)
        base_key = split_keys[0]
        filter_keys = split_keys[1:]

        batched_params = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_repeats,) + jnp.shape(x)),
            par,
        )
        batched_obs = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_repeats,) + jnp.shape(x)),
            obs,
        )
        batched_cond = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_repeats,) + jnp.shape(x)),
            cond_path,
        )

        lm_rec = log_marginal()
        _, _, _, (log_mp,) = vmapped_run_filter(
            bpf,
            filter_keys,
            batched_params,
            batched_obs,
            batched_cond,
            initial_conditions=init_conds,
            observation_history=(),
            recorders=(lm_rec,),
        )

        log_mp = jnp.cumsum(log_mp, axis=1)
        mp_estimates = jnp.asarray(log_mp[:, -1])

        ax.hist(mp_estimates, bins=20, alpha=0.7)
        ax.set_title(f"beta={float(beta):.2f}")

    fig.suptitle("Bootstrap log marginal estimates")
    plt.tight_layout()
    plt.show()
