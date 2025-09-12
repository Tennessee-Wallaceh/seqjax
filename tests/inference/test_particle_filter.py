import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model.linear_gaussian import LinearGaussianSSM, LGSSMParameters
from seqjax.model.simulate import simulate
from seqjax.inference.particlefilter.filter_definitions import BootstrapParticleFilter
from seqjax.inference.particlefilter.base import run_filter
from seqjax.inference.particlefilter.recorders import (
    current_particle_mean,
    current_particle_variance,
    log_marginal,
)
from seqjax.inference.kalman import run_kalman_filter


def _setup(T: int = 40, rho: float = 0.7):
    params = LGSSMParameters(
        transition_matrix=jnp.array([[rho]]),
        transition_noise_scale=jnp.array([1.0]),
        emission_matrix=jnp.array([[1.0]]),
        emission_noise_scale=jnp.array([0.5]),
    )
    model = LinearGaussianSSM()
    key = jrandom.PRNGKey(0)
    _latents, obs, _lat_hist, _obs_hist = simulate(key, model, None, params, T)
    kf_means, kf_covs, kf_logm = run_kalman_filter(params, obs)
    return model, params, obs, kf_means, kf_covs, kf_logm


def test_filtering_moments_close_to_kalman() -> None:
    model, params, obs, kf_means, kf_covs, _ = _setup()
    num_particles = 200
    pf = BootstrapParticleFilter(model, num_particles=num_particles)
    key = jrandom.PRNGKey(1)
    _, _, (pf_means, pf_vars, _) = run_filter(
        pf,
        key,
        params,
        obs,
        recorders=(
            current_particle_mean(lambda p: p.x),
            current_particle_variance(lambda p: p.x),
            log_marginal(),
        ),
    )
    pf_means = jnp.squeeze(jnp.asarray(pf_means))
    pf_vars = jnp.squeeze(jnp.asarray(pf_vars))
    kf_means = jnp.squeeze(kf_means, axis=-1)
    kf_vars = jnp.squeeze(kf_covs)
    mean_err = jnp.mean(jnp.abs(pf_means - kf_means))
    var_err = jnp.mean(jnp.abs(pf_vars - kf_vars))
    rho = params.transition_matrix[0, 0]
    state_sd = params.transition_noise_scale[0] / jnp.sqrt(1.0 - rho**2)
    c = 4.0 * state_sd
    assert mean_err <= c / jnp.sqrt(num_particles)
    assert var_err <= c / jnp.sqrt(num_particles)


def test_log_likelihood_unbiased() -> None:
    model, params, obs, _kf_means, _kf_covs, kf_logm = _setup()
    num_particles = 200
    pf = BootstrapParticleFilter(model, num_particles=num_particles)
    repeats = 20
    key = jrandom.PRNGKey(2)
    pf_lls = []
    for i in range(repeats):
        _, _, (log_incs,) = run_filter(
            pf,
            jrandom.fold_in(key, i),
            params,
            obs,
            recorders=(log_marginal(),),
        )
        pf_lls.append(jnp.sum(jnp.asarray(log_incs)))
    pf_lls = jnp.asarray(pf_lls)
    pf_mean = pf_lls.mean()
    pf_std = pf_lls.std(ddof=1)
    kf_total = kf_logm[-1]
    stderr = pf_std / jnp.sqrt(repeats)
    assert jnp.abs(pf_mean - kf_total) <= 2.0 * stderr


def test_incremental_normalizers() -> None:
    model, params, obs, _kf_means, _kf_covs, kf_logm = _setup()
    num_particles = 200
    pf = BootstrapParticleFilter(model, num_particles=num_particles)
    key = jrandom.PRNGKey(3)
    _, _, (log_incs,) = run_filter(pf, key, params, obs, recorders=(log_marginal(),))
    log_incs = jnp.asarray(log_incs)
    total_from_prod = jnp.log(jnp.prod(jnp.exp(log_incs)))
    total_from_sum = jnp.sum(log_incs)
    assert jnp.allclose(total_from_prod, total_from_sum)
    kf_inc = kf_logm - jnp.concatenate([jnp.array([0.0]), kf_logm[:-1]])
    inc_err = jnp.mean(jnp.abs(log_incs - kf_inc))
    rho = params.transition_matrix[0, 0]
    state_sd = params.transition_noise_scale[0] / jnp.sqrt(1.0 - rho**2)
    c = 4.0 * state_sd
    assert inc_err <= c / jnp.sqrt(num_particles)
