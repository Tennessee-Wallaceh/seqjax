import jax.random as jrandom

from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model.simulate import simulate
from seqjax.inference.particlefilter import (
    BootstrapParticleFilter,
    run_filter,
)


def test_ar1_bootstrap_filter_runs() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()
    _, observations = simulate(key, target, None, parameters, sequence_length=5)

    filter_key = jrandom.PRNGKey(1)
    bpf = BootstrapParticleFilter(target, num_particles=10)
    log_w, particles, ess, rec = run_filter(bpf, filter_key, parameters, observations)

    assert log_w.shape == (bpf.num_particles,)
    assert ess.shape == (observations.y.shape[0],)
    assert rec == ()
