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


def mean_log_vol(weights, particles):
    current = particles[-1]
    return jnp.sum(current.log_vol * weights)


if __name__ == "__main__":
    steps = 200
    params = LogVolWithSkew(
        std_log_vol=jnp.array(3.2),
        mean_reversion=jnp.array(12.0),
        long_term_vol=jnp.array(0.16),
        skew=jnp.array(0.8),
    )

    dt = jnp.array(1.0 / (256 * 8))
    # condition array must include contexts for the prior as well as each
    # filtering step
    cond = TimeIncrement(dt * jnp.ones(steps + 2))

    key = jrandom.key(0)
    latent, obs, *_ = simulate.simulate(
        key,
        SkewStochasticVol(),
        cond,
        params,
        sequence_length=steps,
    )

    bpf = BootstrapParticleFilter(SkewStochasticVol(), num_particles=500)
    filter_key = jrandom.key(1)
    init_conds = tuple(TimeIncrement(cond.dt[i]) for i in range(2))
    cond_path = TimeIncrement(cond.dt[2:])
    log_w, _, log_mp, ess, (filt_lv,) = run_filter(
        bpf,
        filter_key,
        params,
        obs,
        cond_path,
        initial_conditions=init_conds,
        observation_history=params.reference_emission,
        recorders=(mean_log_vol,),
    )

    t = jnp.arange(filt_lv.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(t, latent.log_vol, label="true")
    plt.plot(t, filt_lv, label="filtered")
    plt.legend()
    plt.title("Log volatility")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 2))
    plt.plot(t, ess)
    plt.title("ESS")
    plt.tight_layout()
    plt.show()
