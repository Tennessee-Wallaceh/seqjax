import matplotlib.pyplot as plt
import functools
import jax
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import simulate
from seqjax.model.stochastic_vol import (
    SkewStochasticVol,
    LogVolWithSkew,
    TimeIncrement,
)
from seqjax.inference.particlefilter import (
    BootstrapParticleFilter,
    current_particle_quantiles,
    run_filter,
    log_marginal,
    effective_sample_size,
)


def mean_log_vol(filter_data):
    weights = jax.nn.softmax(filter_data.log_w)
    current = filter_data.particles[-1]
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

    bpf = BootstrapParticleFilter(target, num_particles=500)
    filter_key = jrandom.key(1)
    quant_rec = functools.partial(current_particle_quantiles, quantiles=(0.05, 0.95))
    init_conds = tuple(TimeIncrement(cond.dt[i]) for i in range(target.prior.order))
    cond_path = TimeIncrement(cond.dt[target.prior.order:])

    lm_rec = log_marginal
    ess_rec = effective_sample_size
    log_w, _, _, (log_mp, ess, filt_lv, filt_quant_state) = run_filter(
        bpf,
        filter_key,
        params,
        obs,
        cond_path,
        initial_conditions=init_conds,
        observation_history=(),
        recorders=(lm_rec, ess_rec, mean_log_vol, quant_rec),
    )

    filt_quant = filt_quant_state.log_vol

    t = jnp.arange(filt_lv.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(t, latent.log_vol, label="true")
    plt.plot(t, filt_lv, label="filtered")
    plt.fill_between(t, filt_quant[:, 0], filt_quant[:, 1], alpha=0.3, label="5%-95% quantile")
    plt.legend()
    plt.title("Log volatility")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 2))
    plt.plot(t, ess)
    plt.title("ESS")
    plt.tight_layout()
    plt.show()
