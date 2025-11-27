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


def _setup(T: int = 10, rho: float = 0.7):
    params = LGSSMParameters(
        transition_matrix=jnp.array([[rho]]),
        transition_noise_scale=jnp.array([1.0]),
        emission_matrix=jnp.array([[1.0]]),
        emission_noise_scale=jnp.array([0.5]),
    )
    model = LinearGaussianSSM()
    key = jrandom.PRNGKey(0)
    _latents, obs = simulate(
        key,
        model,
        parameters=params,
        sequence_length=T,
        condition=None,
    )
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
            current_particle_mean,
            current_particle_variance,
            log_marginal,
        ),
    )
    pf_means = jnp.squeeze(jnp.asarray(pf_means.x))
    pf_vars = jnp.squeeze(jnp.asarray(pf_vars.x))
    kf_means = jnp.squeeze(kf_means, axis=-1)
    kf_vars = jnp.squeeze(kf_covs)
    mean_err = jnp.mean(jnp.abs(pf_means - kf_means))
    var_err = jnp.mean(jnp.abs(pf_vars - kf_vars))
    rho = params.transition_matrix[0, 0]
    state_sd = params.transition_noise_scale[0] / jnp.sqrt(1.0 - rho**2)
    c = 4.0 * state_sd
    assert mean_err <= c / jnp.sqrt(num_particles)
    assert var_err <= c / jnp.sqrt(num_particles)


def test_marginal_likelihood_unbiased() -> None:
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
            recorders=(log_marginal,),
        )
        pf_lls.append(jnp.sum(jnp.asarray(log_incs)))
    pf_lls = jnp.asarray(pf_lls, dtype=jnp.float64)

    R = pf_lls.shape[0]
    m = jnp.max(pf_lls)  # max-trick anchor

    u = jnp.exp(pf_lls - m)  # <= 1, stable
    s1 = u.mean()  # ≈ E[exp(L - m)]
    s2 = (u * u).mean()  # ≈ E[exp(2(L - m))]

    log_Z_bar = m + jnp.log(s1)  # log of sample mean \hat Z
    Z_bar = jnp.exp(log_Z_bar)

    # unbiased sample variance of \hat Z, still stable
    var_Z = jnp.exp(2 * m) * (R / (R - 1)) * jnp.maximum(s2 - s1**2, 0.0)
    stderr = jnp.sqrt(var_Z / R)

    Z_true = jnp.exp(kf_logm[-1])
    assert jnp.abs(Z_bar - Z_true) <= 2.0 * stderr


def test_incremental_normalizers() -> None:
    model, params, obs, _kf_means, _kf_covs, kf_logm = _setup()
    num_particles = 200
    pf = BootstrapParticleFilter(model, num_particles=num_particles)
    key = jrandom.PRNGKey(3)
    _, _, (log_incs,) = run_filter(pf, key, params, obs, recorders=(log_marginal,))
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
