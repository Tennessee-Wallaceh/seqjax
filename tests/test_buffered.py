import jax.random as jrandom

from seqjax.model.ar import AR1Target, ARParameters
from seqjax import simulate, BootstrapParticleFilter
from seqjax.inference.buffered import BufferedConfig, run_buffered_filter


def test_buffered_filter_shape() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    params = ARParameters()
    _, obs, _, _ = simulate.simulate(key, target, None, params, sequence_length=5)

    pf = BootstrapParticleFilter(target, num_particles=4)
    config = BufferedConfig(buffer_size=1, batch_size=2, particle_filter=pf)
    log_mps = run_buffered_filter(target, jrandom.PRNGKey(1), params, obs, config=config)

    assert log_mps.shape == (obs.y.shape[0],)
