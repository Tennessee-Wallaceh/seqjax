import jax
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.particlefilter import BootstrapParticleFilter, run_filter
from seqjax.model.logistic_smc import (
    LogisticRegressionSMC,
    LogRegData,
    AnnealCondition,
    DummyObservation,
)


def test_logistic_smc_runs() -> None:
    key = jrandom.PRNGKey(0)
    n, d = 5, 2
    k1, k2, k3 = jrandom.split(key, 3)
    theta_true = jrandom.normal(k1, shape=(d,))
    X = jrandom.normal(k2, shape=(n, d))
    logits = X @ theta_true
    y = jrandom.bernoulli(k3, jax.nn.sigmoid(logits))
    data = LogRegData(X=X, y=y)

    betas = jnp.linspace(0.0, 1.0, 4)
    cond_path = AnnealCondition(beta=betas[1:], beta_prev=betas[:-1])
    obs_path = DummyObservation(dummy=jnp.zeros(len(betas) - 1))

    model = LogisticRegressionSMC()
    pf = BootstrapParticleFilter(model, num_particles=20)
    log_w, particles, log_mp, ess, _ = run_filter(
        pf,
        jrandom.PRNGKey(1),
        data,
        observation_path=obs_path,
        condition_path=cond_path,
        initial_conditions=(AnnealCondition(beta=betas[0], beta_prev=betas[0]),),
    )

    assert log_w.shape == (pf.num_particles,)
    seq_len = betas.shape[0] - 1
    assert log_mp.shape[0] == seq_len
    assert ess.shape[0] == seq_len
