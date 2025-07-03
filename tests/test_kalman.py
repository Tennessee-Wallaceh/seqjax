import jax
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.kalman import run_kalman_filter
from seqjax.inference.particlefilter import (
    BootstrapParticleFilter,
    run_filter,
    current_particle_mean,
    current_particle_variance,
)
from seqjax import simulate
from seqjax.model.linear_gaussian import LinearGaussianSSM, LGSSMParameters


def test_kalman_filter_matches_particle_filter() -> None:
    seq_len = 6
    key = jrandom.PRNGKey(0)
    target = LinearGaussianSSM()
    params = LGSSMParameters()
    _, obs, _, _ = simulate.simulate(key, target, None, params, sequence_length=seq_len)

    pf = BootstrapParticleFilter(target, num_particles=1000)
    mean_rec = current_particle_mean(lambda p: p.x)
    var_rec = current_particle_variance(lambda p: p.x)
    log_w, _, _, _, (mean_hist, var_hist) = run_filter(
        pf,
        jrandom.PRNGKey(1),
        params,
        obs,
        initial_conditions=(None,),
        recorders=(mean_rec, var_rec),
    )

    kf_mean, kf_cov, _ = run_kalman_filter(params, obs)

    assert kf_mean.shape == mean_hist.shape
    assert kf_cov.shape[:2] == (seq_len, params.transition_matrix.shape[0])
    assert jnp.allclose(kf_mean, mean_hist, atol=1e-1, rtol=1e-1)
    assert jnp.allclose(jnp.diagonal(kf_cov, axis1=1, axis2=2), var_hist, atol=1e-1, rtol=1e-1)
