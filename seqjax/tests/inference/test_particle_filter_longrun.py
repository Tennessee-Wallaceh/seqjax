"""Long-running statistical validations for the particle filter."""

from typing import Any, Tuple

import pytest
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array

from seqjax.inference.kalman import run_kalman_filter
from seqjax.inference.particlefilter.base import run_filter
from seqjax.inference.particlefilter.filter_definitions import BootstrapParticleFilter
from seqjax.inference.particlefilter.recorders import (
    current_particle_mean,
    current_particle_variance,
    log_marginal,
)
from seqjax.model.linear_gaussian import LinearGaussianSSM, LGSSMParameters
from seqjax.model.simulate import simulate


def _setup(T: int = 10, rho: float = 0.7) -> Tuple[
    LinearGaussianSSM,
    LGSSMParameters,
    Any,
    Any,
    Any,
    Any,
]:
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


_NUM_PARTICLES = 2_000
_REPEATS = 200
_T = 30


@pytest.mark.longrun
def test_filtering_moments_close_to_kalman_longrun() -> None:
    model, params, obs, kf_means, kf_covs, _ = _setup(T=_T)
    pf = BootstrapParticleFilter(model, num_particles=_NUM_PARTICLES)
    key = jrandom.PRNGKey(101)
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
    )  # type: ignore[misc]

    pf_means = jnp.squeeze(jnp.asarray(pf_means.x))
    pf_vars = jnp.squeeze(jnp.asarray(pf_vars.x))
    kf_means = jnp.squeeze(kf_means, axis=-1)
    kf_vars = jnp.squeeze(kf_covs)

    mean_err = jnp.mean(jnp.abs(pf_means - kf_means))
    var_err = jnp.mean(jnp.abs(pf_vars - kf_vars))

    rho = params.transition_matrix[0, 0]
    state_sd = params.transition_noise_scale[0] / jnp.sqrt(1.0 - rho**2)
    c = 3.0 * state_sd
    assert mean_err <= c / jnp.sqrt(_NUM_PARTICLES)
    assert var_err <= c / jnp.sqrt(_NUM_PARTICLES)


@pytest.mark.longrun
def test_marginal_likelihood_unbiased_longrun() -> None:
    model, params, obs, _kf_means, _kf_covs, kf_logm = _setup(T=_T)
    pf = BootstrapParticleFilter(model, num_particles=_NUM_PARTICLES)
    key = jrandom.PRNGKey(202)
    pf_lls_raw: list[Array] = []
    for i in range(_REPEATS):
        _, _, (log_incs,) = run_filter(
            pf,
            jrandom.fold_in(key, i),
            params,
            obs,
            recorders=(log_marginal,),
        )  # type: ignore[misc]
        pf_lls_raw.append(jnp.sum(jnp.asarray(log_incs)))
    pf_lls = jnp.asarray(pf_lls_raw, dtype=jnp.float64)

    r = pf_lls.shape[0]
    m = jnp.max(pf_lls)

    u = jnp.exp(pf_lls - m)
    s1 = u.mean()
    s2 = (u * u).mean()

    log_z_bar = m + jnp.log(s1)
    z_bar = jnp.exp(log_z_bar)

    var_z = jnp.exp(2 * m) * (r / (r - 1)) * jnp.maximum(s2 - s1**2, 0.0)
    stderr = jnp.sqrt(var_z / r)

    z_true = jnp.exp(kf_logm[-1])
    tolerance = jnp.maximum(1.5 * stderr, jnp.finfo(jnp.float32).eps)
    assert jnp.abs(z_bar - z_true) <= tolerance


@pytest.mark.longrun
def test_incremental_normalizers_longrun() -> None:
    model, params, obs, _kf_means, _kf_covs, kf_logm = _setup(T=_T)
    pf = BootstrapParticleFilter(model, num_particles=_NUM_PARTICLES)
    key = jrandom.PRNGKey(303)
    _, _, (log_incs,) = run_filter(  # type: ignore[misc]
        pf, key, params, obs, recorders=(log_marginal,)
    )

    log_incs = jnp.asarray(log_incs)
    total_from_prod = jnp.log(jnp.prod(jnp.exp(log_incs)))
    total_from_sum = jnp.sum(log_incs)
    assert jnp.allclose(total_from_prod, total_from_sum)

    kf_inc = kf_logm - jnp.concatenate([jnp.array([0.0]), kf_logm[:-1]])
    inc_err = jnp.mean(jnp.abs(log_incs - kf_inc))
    rho = params.transition_matrix[0, 0]
    state_sd = params.transition_noise_scale[0] / jnp.sqrt(1.0 - rho**2)
    c = 3.0 * state_sd
    assert inc_err <= c / jnp.sqrt(_NUM_PARTICLES)
