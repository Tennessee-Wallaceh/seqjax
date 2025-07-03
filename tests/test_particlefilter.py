import jax.random as jrandom
import pytest

from seqjax.model.ar import AR1Target, ARParameters
from seqjax import BootstrapParticleFilter, simulate
from seqjax.inference.particlefilter import run_filter


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
