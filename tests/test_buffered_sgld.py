import jax.random as jrandom

from seqjax.model.ar import AR1Target, ARParameters, HalfCauchyStds
from seqjax import simulate, BootstrapParticleFilter
from seqjax.inference.buffered import BufferedSGLDConfig, run_buffered_sgld


def test_buffered_sgld_runs() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    params = ARParameters()
    prior = HalfCauchyStds()
    _, obs, _, _ = simulate.simulate(key, target, None, params, sequence_length=5)

    pf = BootstrapParticleFilter(target, num_particles=4)
    config = BufferedSGLDConfig(
        step_size=0.1,
        num_iters=3,
        buffer_size=1,
        batch_size=2,
        particle_filter=pf,
        parameter_prior=prior,
    )
    samples = run_buffered_sgld(target, jrandom.PRNGKey(1), params, obs, config=config)

    assert samples.ar.shape == (config.num_iters,)
