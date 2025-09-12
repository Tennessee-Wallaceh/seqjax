import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import simulate
from seqjax.model import ar
from seqjax.inference.mcmc import nuts

if __name__ == "__main__":
    true_params = ar.ARParameters(
        ar=jnp.array(0.8),
        observation_std=jnp.array(0.5),
        transition_std=jnp.array(0.5),
    )
    target_posterior = ar.AR1Bayesian(true_params)
    sequence_length = 50

    key = jrandom.PRNGKey(0)
    latents, obs, _, _ = simulate.simulate(
        key, target_posterior.target, None, true_params, sequence_length=sequence_length
    )

    config = nuts.NUTSConfig(
        step_size=1e-5, num_warmup=5000, num_samples=2000, num_chains=2
    )

    time_s, latent_samples, parameter_samples = nuts.run_bayesian_nuts(
        target_posterior=target_posterior,
        hyperparameters=None,
        key=key,
        observation_path=obs,
        condition_path=None,
        config=config,
    )

    ar_samples = jnp.asarray(parameter_samples.ar)

    # latent_samples = jnp.asarray(latent_samples)
    # mean_latent = jnp.mean(latent_samples, axis=0)
    # lower_latent = jnp.quantile(latent_samples, 0.05, axis=0)
    # upper_latent = jnp.quantile(latent_samples, 0.95, axis=0)

    # t = jnp.arange(sequence_length)
    # plt.figure(figsize=(6, 4))
    # plt.plot(t, latents.x, label="true latent")
    # plt.plot(t, mean_latent, label="mean sample")
    # plt.fill_between(t, lower_latent, upper_latent, color="gray", alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(6, 3))
    plt.plot(ar_samples)
    plt.axhline(true_params.ar, color="r", linestyle="--", label="true")
    plt.axhline(jnp.mean(ar_samples), color="g", linestyle="--", label="mean")
    plt.xlabel("iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()

    q05, q95 = jnp.quantile(ar_samples, jnp.array([0.05, 0.95]))
    plt.figure(figsize=(6, 3))
    plt.hist(ar_samples, bins=30, density=True, alpha=0.7)
    plt.axvline(true_params.ar, color="r", linestyle="--", label="true")
    plt.axvline(q05, color="k", linestyle="-.", label="q05")
    plt.axvline(q95, color="k", linestyle="-.", label="q95")
    plt.legend()
    plt.tight_layout()
    plt.show()
