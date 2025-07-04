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
    current_particle_quantiles,
    run_filter,
)
from seqjax.inference.autoregressive_vi import (
    RandomAutoregressor,
    AutoregressiveVIConfig,
    run_autoregressive_vi,
)
from seqjax.inference.embedder import PassThroughEmbedder
from seqjax.model.base import ParameterPrior, HyperParameters


class AROnlyPrior(ParameterPrior[ARParameters, HyperParameters]):
    """Uniform prior over the AR coefficient only."""

    @staticmethod
    def sample(key: jrandom.PRNGKey, _hyper: HyperParameters) -> ARParameters:
        return ARParameters(
            ar=jrandom.uniform(key, minval=-1.0, maxval=1.0),
            observation_std=jnp.array(0.05),
            transition_std=jnp.array(0.01),
        )

    @staticmethod
    def log_prob(params: ARParameters, _hyper: HyperParameters) -> jnp.ndarray:
        return jstats.uniform.logpdf(params.ar, loc=-1.0, scale=2.0)


if __name__ == "__main__":
    target = AR1Target()
    true_params = ARParameters(
        ar=jnp.array(0.8), observation_std=jnp.array(0.05), transition_std=jnp.array(0.01)
    )
    key = jrandom.PRNGKey(0)
    latents, obs, _, _ = simulate.simulate(key, target, None, true_params, sequence_length=100)

    pf = BootstrapParticleFilter(target, num_particles=256)
    prior = AROnlyPrior()

    # MCMC using NUTS
    nuts_cfg = NUTSConfig(num_warmup=500, num_samples=1000, step_size=0.03)
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

    # Particle MCMC
    pmcmc_cfg = ParticleMCMCConfig(
        mcmc=RandomWalkConfig(step_size=0.005, num_samples=1000),
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

    # Buffered SGMCMC
    sgld_cfg = BufferedSGLDConfig(
        step_size=ARParameters(ar=jnp.array(5e-4), observation_std=0.0, transition_std=0.0),
        num_iters=1000,
        buffer_size=1,
        batch_size=10,
        particle_filter=pf,
        parameter_prior=prior,
    )
    sgld_key = jrandom.PRNGKey(3)
    sgld_params = run_buffered_sgld(target, sgld_key, true_params, obs, config=sgld_cfg)
    sgld_ar = jnp.asarray(sgld_params.ar)

    # Autoregressive VI for latent paths
    embedder = PassThroughEmbedder(sample_length=obs.y.shape[0], prev_window=0, post_window=0)
    sampler = RandomAutoregressor(
        sample_length=obs.y.shape[0],
        x_dim=1,
        context_dim=embedder.context_dimension,
        parameter_dim=1,
        lag_order=1,
    )
    vi_cfg = AutoregressiveVIConfig(sampler=sampler, embedder=embedder, num_samples=3)
    vi_key = jrandom.PRNGKey(4)
    vi_latents = run_autoregressive_vi(
        target,
        vi_key,
        obs,
        parameters=true_params,
        config=vi_cfg,
    )

    # Quantiles for PMCMC and SGLD
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
    sgld_mean = ARParameters(
        ar=jnp.mean(sgld_ar),
        observation_std=true_params.observation_std,
        transition_std=true_params.transition_std,
    )
    _, _, _, _, (sgld_quant,) = run_filter(
        pf,
        jrandom.PRNGKey(6),
        sgld_mean,
        obs,
        initial_conditions=(None,),
        recorders=(quant_rec,),
    )

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex="col")

    axes[0, 0].hist(nuts_ar, bins=30, density=True)
    axes[0, 0].axvline(true_params.ar, color="r", linestyle="--")
    axes[0, 0].set_title("NUTS AR posterior")

    axes[1, 0].hist(pmcmc_ar, bins=30, density=True)
    axes[1, 0].axvline(true_params.ar, color="r", linestyle="--")
    axes[1, 0].set_title("PMCMC AR posterior")

    axes[2, 0].hist(sgld_ar, bins=30, density=True)
    axes[2, 0].axvline(true_params.ar, color="r", linestyle="--")
    axes[2, 0].set_title("SGLD AR posterior")

    axes[3, 0].axis("off")
    axes[3, 0].text(0.5, 0.5, "Autoregressive\nN/A", ha="center", va="center")

    axes[0, 1].plot(nuts_latents.x.T[:3].T, alpha=0.7)
    axes[0, 1].set_title("NUTS latent samples")

    axes[1, 1].fill_between(
        jnp.arange(obs.y.shape[0]),
        pmcmc_quant[:, 0],
        pmcmc_quant[:, 1],
        color="gray",
        alpha=0.5,
    )
    axes[1, 1].set_title("PMCMC particle quantiles")

    axes[2, 1].fill_between(
        jnp.arange(obs.y.shape[0]),
        sgld_quant[:, 0],
        sgld_quant[:, 1],
        color="gray",
        alpha=0.5,
    )
    axes[2, 1].set_title("SGLD particle quantiles")

    axes[3, 1].plot(vi_latents.x[0], alpha=0.7)
    axes[3, 1].set_title("Autoregressive sample")

    for ax in axes[:, 0]:
        ax.set_xlim(-1, 1)
    plt.tight_layout()
    plt.show()
