import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import simulate
from seqjax.model.ar import AR1Target, ARParameters, HalfCauchyStds
from seqjax.inference.particlefilter import (
    BootstrapParticleFilter,
    current_particle_mean,
    current_particle_quantiles,
    run_filter,
    log_marginal,
)
from seqjax.inference.buffered import BufferedSGLDConfig, run_buffered_sgld
from seqjax.inference.sgld import SGLDConfig
from seqjax.inference.mcmc import NUTSConfig, run_bayesian_nuts


if __name__ == "__main__":
    # generate data from a simple AR(1) model
    seq_len = 100
    true_params = ARParameters(ar=jnp.array(0.8))
    key = jrandom.PRNGKey(0)
    latents, obs, _, _ = simulate.simulate(
        key, AR1Target(), None, true_params, sequence_length=seq_len
    )

    pf = BootstrapParticleFilter(AR1Target(), num_particles=500)
    prior = HalfCauchyStds()

    # NUTS posterior (baseline)
    print("NUTS")
    nuts_latents, nuts_params = run_bayesian_nuts(
        AR1Target(),
        jrandom.PRNGKey(1),
        obs,
        parameter_prior=prior,
        initial_latents=latents,
        initial_parameters=true_params,
        config=NUTSConfig(step_size=0.005, num_warmup=1000, num_samples=10000),
    )

    # Full sequence score SGLD
    print("SGLD full")
    sgld_full = run_buffered_sgld(
        AR1Target(),
        jrandom.PRNGKey(2),
        ARParameters(ar=jnp.array(0.0)),
        obs,
        config=BufferedSGLDConfig(
            buffer_size=0,
            batch_size=seq_len,
            particle_filter=pf,
            parameter_prior=prior,
        ),
        sgld_config=SGLDConfig(
            step_size=ARParameters(ar=jnp.array(1e-4)), num_iters=10000
        ),
    )

    # Small buffer SGLD
    print("SGLD small")
    sgld_small = run_buffered_sgld(
        AR1Target(),
        jrandom.PRNGKey(3),
        ARParameters(ar=jnp.array(0.0)),
        obs,
        config=BufferedSGLDConfig(
            buffer_size=1,
            batch_size=10,
            particle_filter=pf,
            parameter_prior=prior,
        ),
        sgld_config=SGLDConfig(
            step_size=ARParameters(ar=jnp.array(1e-4)), num_iters=10000
        ),
    )

    # Large buffer SGLD
    # print("SGLD large")
    # sgld_large = run_buffered_sgld(
    #     AR1Target(),
    #     jrandom.PRNGKey(4),
    #     ARParameters(ar=jnp.array(0.0)),
    #     obs,
    #     config=BufferedSGLDConfig(
    #         buffer_size=10,
    #         batch_size=10,
    #         particle_filter=pf,
    #         parameter_prior=prior,
    #     ),
    #     sgld_config=SGLDConfig(
    #         step_size=ARParameters(ar=jnp.array(1e-4)), num_iters=10000
    #     ),
    # )

    nuts_ar = nuts_params.ar
    full_ar = sgld_full.ar
    small_ar = sgld_small.ar
    large_ar = sgld_large.ar

    sample_sets = [
        ("NUTS", nuts_ar),
        # ("FULL SGLD", full_ar),
        # ("SMALL SGLD", small_ar),
        # ("LARGE SGLD", large_ar),
    ]
    for label, ar_set in sample_sets:
        q05, q95 = jnp.quantile(ar_set, jnp.array([0.05, 0.95]))
        print(f"{label}: {jnp.mean(ar_set):.2f} ({q05:.2f}, {q95:.2f})")

    # histogram comparison
    all_samples = jnp.concatenate([nuts_ar, full_ar, small_ar, large_ar])
    bins = jnp.linspace(jnp.min(all_samples), jnp.max(all_samples), 40)
    plt.figure(figsize=(6, 3))
    plt.hist(nuts_ar, bins=bins, density=True, alpha=0.5, label="NUTS")
    plt.hist(full_ar, bins=bins, density=True, alpha=0.5, label="full score")
    plt.hist(small_ar, bins=bins, density=True, alpha=0.5, label="buffer=1")
    plt.hist(large_ar, bins=bins, density=True, alpha=0.5, label="buffer=10")
    plt.axvline(true_params.ar, color="k", linestyle="--", label="true")
    plt.xlabel("ar parameter")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # trace plots
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(nuts_ar)
    axes[0].set_ylabel("NUTS")
    axes[1].plot(full_ar)
    axes[1].set_ylabel("full score")
    axes[2].plot(small_ar)
    axes[2].set_ylabel("buffer=1")
    axes[3].plot(large_ar)
    axes[3].set_ylabel("buffer=10")
    for ax in axes:
        ax.axhline(true_params.ar, color="r", linestyle="--")
    plt.xlabel("iteration")
    plt.tight_layout()
    plt.show()

    # latent posterior summaries
    def latent_summary(params, rng):
        mean_rec = current_particle_mean(lambda p: p.x)
        quant_rec = current_particle_quantiles(lambda p: p.x, quantiles=(0.05, 0.95))
        _, _, _, (_, mean_lat, quant_lat) = run_filter(
            pf,
            rng,
            params,
            obs,
            recorders=(log_marginal(), mean_rec, quant_rec),
            initial_conditions=(),
        )
        return mean_lat, jnp.transpose(quant_lat)

    nuts_mean = jnp.mean(nuts_latents.x, axis=0)
    nuts_q = jnp.quantile(nuts_latents.x, jnp.array([0.05, 0.95]), axis=0)
    full_mean, full_q = latent_summary(ARParameters(ar=full_ar[-1]), jrandom.PRNGKey(5))
    small_mean, small_q = latent_summary(
        ARParameters(ar=small_ar[-1]), jrandom.PRNGKey(6)
    )
    large_mean, large_q = latent_summary(
        ARParameters(ar=large_ar[-1]), jrandom.PRNGKey(7)
    )

    t = jnp.arange(seq_len)
    plt.figure(figsize=(8, 4))
    plt.plot(t, latents.x, label="true")
    plt.plot(t, nuts_mean, label="NUTS")
    plt.fill_between(t, nuts_q[0], nuts_q[1], alpha=0.3)
    plt.plot(t, full_mean, label="full score")
    plt.fill_between(t, full_q[0], full_q[1], alpha=0.3)
    plt.plot(t, small_mean, label="buffer=1")
    plt.fill_between(t, small_q[0], small_q[1], alpha=0.3)
    plt.plot(t, large_mean, label="buffer=10")
    plt.fill_between(t, large_q[0], large_q[1], alpha=0.3)
    plt.xlabel("time")
    plt.ylabel("latent x")
    plt.legend()
    plt.tight_layout()
    plt.show()
