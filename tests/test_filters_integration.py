import jax.random as jrandom
import jax.numpy as jnp
import pytest

from seqjax import BootstrapParticleFilter, AuxiliaryParticleFilter, simulate
from seqjax.inference.particlefilter import run_filter
from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model.linear_gaussian import LinearGaussianSSM, LGSSMParameters
from seqjax.model.stochastic_vol import (
    SimpleStochasticVol,
    LogVolRW,
    TimeIncrement,
    LogReturnObs,
)
from seqjax.model.poisson_ssm import PoissonSSM, PoissonSSMParameters


@pytest.mark.parametrize("filter_cls", [BootstrapParticleFilter, AuxiliaryParticleFilter])
def test_filters_ar1_and_stochastic_vol(filter_cls) -> None:
    seq_len = 5

    # AR(1) model
    key = jrandom.PRNGKey(0)
    ar_target = AR1Target()
    ar_params = ARParameters()
    _, ar_obs, _, _ = simulate.simulate(key, ar_target, None, ar_params, sequence_length=seq_len)
    filter_key = jrandom.PRNGKey(1)
    ar_pf = filter_cls(ar_target, num_particles=5)
    log_w, _, _, _, _ = run_filter(
        ar_pf,
        filter_key,
        ar_params,
        ar_obs,
        initial_conditions=(None,),
    )
    assert log_w.shape == (ar_pf.num_particles,)

    # Stochastic volatility model
    key = jrandom.PRNGKey(2)
    sv_target = SimpleStochasticVol()
    sv_params = LogVolRW(
        std_log_vol=jnp.array(0.1),
        mean_reversion=jnp.array(0.1),
        long_term_vol=jnp.array(1.0),
    )
    full_cond = TimeIncrement(jnp.ones(seq_len + sv_target.prior.order - 1))
    _, sv_obs, _, _ = simulate.simulate(
        key, sv_target, full_cond, sv_params, sequence_length=seq_len
    )
    sv_pf = filter_cls(sv_target, num_particles=5)
    log_w, _, _, _, _ = run_filter(
        sv_pf,
        jrandom.PRNGKey(3),
        sv_params,
        sv_obs,
        condition_path=TimeIncrement(full_cond.dt[sv_target.prior.order - 1 :]),
        initial_conditions=tuple(
            TimeIncrement(full_cond.dt[i]) for i in range(sv_target.prior.order)
        ),
        observation_history=()
    )
    assert log_w.shape == (sv_pf.num_particles,)


@pytest.mark.parametrize("filter_cls", [BootstrapParticleFilter, AuxiliaryParticleFilter])
def test_filters_linear_gaussian(filter_cls) -> None:
    seq_len = 4

    key = jrandom.PRNGKey(4)
    target = LinearGaussianSSM()
    params = LGSSMParameters()
    _, obs, _, _ = simulate.simulate(key, target, None, params, sequence_length=seq_len)
    pf = filter_cls(target, num_particles=5)
    log_w, _, _, _, _ = run_filter(
        pf,
        jrandom.PRNGKey(5),
        params,
        obs,
        initial_conditions=(None,),
    )
    assert log_w.shape == (pf.num_particles,)


@pytest.mark.parametrize("filter_cls", [BootstrapParticleFilter, AuxiliaryParticleFilter])
def test_filters_poisson_ssm(filter_cls) -> None:
    seq_len = 5

    key = jrandom.PRNGKey(6)
    target = PoissonSSM()
    params = PoissonSSMParameters()
    _, obs, _, _ = simulate.simulate(key, target, None, params, sequence_length=seq_len)
    pf = filter_cls(target, num_particles=5)
    log_w, _, _, _, _ = run_filter(
        pf,
        jrandom.PRNGKey(7),
        params,
        obs,
        initial_conditions=(None,),
    )
    assert log_w.shape == (pf.num_particles,)
