import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.particlefilter import (
    BootstrapParticleFilter,
    current_particle_mean,
    current_particle_quantiles,
    run_filter,
    log_marginal,
    effective_sample_size,
)
from seqjax.model.logistic_smc import (
    LogisticRegressionSMC,
    LogRegData,
    AnnealCondition,
    DummyObservation,
)


if __name__ == "__main__":
    # simulate a 2d logistic regression dataset
    n, d = 200, 2
    key = jrandom.key(0)
    k1, k2, k3 = jrandom.split(key, 3)
    theta_true = jrandom.normal(k1, shape=(d,))
    X = jrandom.normal(k2, shape=(n, d))
    logits = X @ theta_true
    y = jrandom.bernoulli(k3, jax.nn.sigmoid(logits))
    data = LogRegData(X=X, y=y)

    # annealing schedule from beta=0 -> 1
    num_steps = 20
    betas = jnp.linspace(0.0, 1.0, num_steps + 1)
    cond_path = AnnealCondition(beta=betas[1:], beta_prev=betas[:-1])
    obs_path = DummyObservation(dummy=jnp.zeros(num_steps))

    model = LogisticRegressionSMC()
    pf = BootstrapParticleFilter(model, num_particles=1000)
    init_cond = (AnnealCondition(beta=betas[0], beta_prev=betas[0]),)

    mean_rec = current_particle_mean(lambda p: p.theta)
    quant_rec = current_particle_quantiles(lambda p: p.theta, quantiles=(0.05, 0.95))

    lm_rec = log_marginal()
    ess_rec = effective_sample_size()
    log_w, particles, _, (log_mp, ess, theta_mean, theta_quant) = run_filter(
        pf,
        jrandom.key(1),
        data,
        obs_path,
        cond_path,
        initial_conditions=init_cond,
        recorders=(lm_rec, ess_rec, mean_rec, quant_rec),
    )

    weights = jax.nn.softmax(log_w)
    theta_samples = particles[-1].theta

    plt.figure(figsize=(6, 5))
    plt.scatter(theta_samples[:, 0], theta_samples[:, 1], c=weights, cmap="viridis", s=10, alpha=0.7)
    plt.scatter(theta_true[0], theta_true[1], color="red", marker="x", label="true")
    plt.xlabel(r"$\\theta_1$")
    plt.ylabel(r"$\\theta_2$")
    plt.colorbar(label="weight")
    plt.title("SMC posterior for logistic regression")
    plt.legend()
    plt.tight_layout()
    plt.show()

    t = jnp.arange(num_steps)
    plt.figure(figsize=(6, 3))
    plt.plot(t, ess)
    plt.title("ESS across annealing steps")
    plt.tight_layout()
    plt.show()
