import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import simulate
from seqjax.model.stochastic_vol import (
    SkewStochasticVol,
    LogVolWithSkew,
    TimeIncrement,
)
from seqjax.inference.particlefilter import BootstrapParticleFilter, run_filter


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
    cond = TimeIncrement(dt * jnp.ones(steps + 2))
    key = jrandom.key(0)
    latent, obs, *_ = simulate.simulate(
        key,
        SkewStochasticVol(),
        cond,
        params,
        sequence_length=steps,
    )

    betas = jnp.array([-0.8, -0.5, -0.2, 0.0])
    num_repeats = 50

    init_conds = tuple(TimeIncrement(cond.dt[i]) for i in range(2))
    cond_path = TimeIncrement(cond.dt[2:])
    bpf = BootstrapParticleFilter(SkewStochasticVol(), num_particles=500)

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

        mp_estimates = []
        for _ in range(num_repeats):
            base_key, filter_key = jrandom.split(base_key)
            _, _, log_mp, _, _ = run_filter(
                bpf,
                filter_key,
                par,
                obs,
                cond_path,
                initial_conditions=init_conds,
                observation_history=params.reference_emission,
            )
            mp_estimates.append(float(log_mp[-1]))

        ax.hist(jnp.array(mp_estimates), bins=20, alpha=0.7)
        ax.set_title(f"beta={float(beta):.2f}")

    fig.suptitle("Bootstrap log marginal estimates")
    plt.tight_layout()
    plt.show()
