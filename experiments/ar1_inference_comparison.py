import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats

from seqjax.model import simulate
from seqjax.model.ar import AR1Target, ARParameters
from seqjax import (
    BootstrapParticleFilter,
    NUTSConfig,
    run_bayesian_nuts,
    RandomWalkConfig,
    ParticleMCMCConfig,
    run_particle_mcmc,
    BufferedSGLDConfig,
    run_buffered_sgld,
)
from seqjax.inference.particlefilter import current_particle_quantiles, run_filter
from seqjax.model.base import ParameterPrior
from seqjax.model.typing import HyperParameters


class AROnlyPrior(ParameterPrior[ARParameters, HyperParameters]):
    """Uniform prior over the AR coefficient only."""

    @staticmethod
    def sample(key: jrandom.PRNGKey, _hyper: HyperParameters) -> ARParameters:
        return ARParameters(
            ar=jrandom.uniform(key, minval=-1.0, maxval=1.0),
            observation_std=jnp.array(0.02),  # known
            transition_std=jnp.array(0.01),  # known
        )

    @staticmethod
    def log_prob(params: ARParameters, _hyper: HyperParameters) -> jnp.ndarray:
        return jstats.uniform.logpdf(params.ar, loc=-1.0, scale=2.0)


if __name__ == "__main__":
    target = AR1Target()
    true_params = ARParameters(
        ar=jnp.array(0.5),
        observation_std=jnp.array(0.1),
        transition_std=jnp.array(1.0),
    )
    key = jrandom.PRNGKey(0)
    latents, obs, _, _ = simulate.simulate(
        key, target, None, true_params, sequence_length=50
    )

    prior = AROnlyPrior()

    # MCMC using NUTS
    print("NUTS")
    nuts_cfg = NUTSConfig(num_warmup=1000, num_samples=1000, step_size=1e-3)
    nuts_key = jrandom.PRNGKey(1)
    nuts_latents, nuts_params = run_bayesian_nuts(
        target,
        nuts_key,
        obs,
        parameter_prior=prior,
        initial_latents=latents,
        initial_parameters=true_params,
        config=nuts_cfg,
    )
    nuts_ar = jnp.asarray(nuts_params.ar)

    plt.figure(figsize=(8, 3))
    plt.title("NUTS ar trace")
    plt.plot(nuts_params.ar)
    plt.grid()

    # # Particle MCMC
    print("Particle MCMC")
    pf = BootstrapParticleFilter(target, num_particles=256)
    pmcmc_cfg = ParticleMCMCConfig(
        mcmc=NUTSConfig(step_size=1e-3, num_samples=1000),
        particle_filter=pf,
    )
    pmcmc_key = jrandom.PRNGKey(2)
    pmcmc_params = run_particle_mcmc(
        target,
        pmcmc_key,
        obs,
        parameter_prior=prior,
        config=pmcmc_cfg,
        initial_parameters=true_params,
        initial_conditions=(None,),
    )
    pmcmc_ar = jnp.asarray(pmcmc_params.ar)

    plt.figure(figsize=(8, 3))
    plt.title("PMCMC ar trace")
    plt.plot(pmcmc_ar)
    plt.grid()

    # Buffered SGMCMC
    # print("buffered SGMCMC")
    # sgld_cfg = BufferedSGLDConfig(
    #     step_size=ARParameters(
    #         ar=jnp.array(3e-4), observation_std=0.0, transition_std=0.0
    #     ),
    #     num_iters=20000,
    #     buffer_size=5,
    #     batch_size=5,
    #     particle_filter=pf,
    #     parameter_prior=prior,
    # )
    # sgld_key = jrandom.PRNGKey(3)
    # sgld_params = run_buffered_sgld(target, sgld_key, true_params, obs, config=sgld_cfg)
    # sgld_ar = jnp.asarray(sgld_params.ar)

    # Quantiles for PMCMC and SGLD
    print("quantiles")
    quant_rec = current_particle_quantiles(lambda p: p.x, quantiles=(0.05, 0.95))
    pmcmc_mean = ARParameters(
        ar=jnp.mean(pmcmc_ar),
        observation_std=true_params.observation_std,
        transition_std=true_params.transition_std,
    )
    _, _, _, _, (pmcmc_quant,) = run_filter(
        pf,
        jrandom.PRNGKey(5),
        pmcmc_mean,
        obs,
        initial_conditions=(None,),
        recorders=(quant_rec,),
    )
    # sgld_mean = ARParameters(
    #     ar=jnp.mean(sgld_ar),
    #     observation_std=true_params.observation_std,
    #     transition_std=true_params.transition_std,
    # )
    # _, _, _, _, (sgld_quant,) = run_filter(
    #     pf,
    #     jrandom.PRNGKey(6),
    #     sgld_mean,
    #     obs,
    #     initial_conditions=(None,),
    #     recorders=(quant_rec,),
    # )

    print("Plots")
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    bins = jnp.arange(-1, 1.01, 0.01)

    axes[0, 0].hist(nuts_ar, bins=bins, density=True)
    axes[0, 0].axvline(true_params.ar, color="r", linestyle="--")
    axes[0, 0].set_title("NUTS AR posterior")

    axes[1, 0].hist(pmcmc_ar, bins=bins, density=True)
    axes[1, 0].axvline(true_params.ar, color="r", linestyle="--")
    axes[1, 0].set_title("PMCMC AR posterior")

    # axes[2, 0].hist(sgld_ar, bins=bins, density=True)
    # axes[2, 0].axvline(true_params.ar, color="r", linestyle="--")
    # axes[2, 0].set_title("SGLD AR posterior")

    axes[0, 1].plot(nuts_latents.x[-3:, :].T, alpha=0.7)
    axes[0, 1].plot(latents.x, color="k", linestyle="--", label="true")
    axes[0, 1].set_title("NUTS latent samples")

    axes[1, 1].fill_between(
        jnp.arange(obs.y.shape[0]),
        pmcmc_quant[:, 0],
        pmcmc_quant[:, 1],
        color="gray",
        alpha=0.5,
    )
    axes[1, 1].plot(latents.x, color="k", linestyle="--")
    axes[1, 1].set_title("PMCMC particle quantiles")

    # axes[2, 1].fill_between(
    #     jnp.arange(obs.y.shape[0]),
    #     sgld_quant[:, 0],
    #     sgld_quant[:, 1],
    #     color="gray",
    #     alpha=0.5,
    # )
    # axes[2, 1].plot(latents.x, color="k", linestyle="--")
    # axes[2, 1].set_title("SGLD particle quantiles")

    for ax in axes[:, 0]:
        ax.set_xlim(-1, 1)
    plt.tight_layout()
    plt.show()
