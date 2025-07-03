import jax.random as jrandom
import pytest

from seqjax.model.ar import AR1Target, ARParameters
from seqjax import BootstrapParticleFilter, simulate
from seqjax.inference.particlefilter import (
    run_filter,
    current_particle_quantiles,
    current_particle_variance,
)


def test_ar1_bootstrap_filter_runs() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )
    filter_key = jrandom.PRNGKey(1)
    bpf = BootstrapParticleFilter(target, num_particles=10)
    log_w, particles, ess, rec = run_filter(
        bpf,
        filter_key,
        parameters,
        observations,
        initial_conditions=(None,),
    )

    assert log_w.shape == (bpf.num_particles,)
    assert ess.shape == (observations.y.shape[0],)
    assert rec == ()


def test_run_filter_requires_initial_conditions() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )
    filter_key = jrandom.PRNGKey(1)
    bpf = BootstrapParticleFilter(target, num_particles=10)

    with pytest.raises(ValueError):
        run_filter(bpf, filter_key, parameters, observations)


def test_particle_recorders_shapes() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )
    filter_key = jrandom.PRNGKey(1)
    bpf = BootstrapParticleFilter(target, num_particles=20)

    quant_rec = current_particle_quantiles(lambda p: p.x)
    var_rec = current_particle_variance(lambda p: p.x)

    log_w, _, _, (quant_hist, var_hist) = run_filter(
        bpf,
        filter_key,
        parameters,
        observations,
        initial_conditions=(None,),
        recorders=(quant_rec, var_rec),
    )

    seq_len = observations.y.shape[0]
    assert quant_hist.shape == (seq_len, 2)
    assert var_hist.shape == (seq_len,)
