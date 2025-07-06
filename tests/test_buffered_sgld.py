import jax.random as jrandom

from seqjax.model.ar import AR1Target, ARParameters, HalfCauchyStds
from seqjax import simulate, BootstrapParticleFilter
from seqjax.inference.buffered import BufferedSGLDConfig, run_buffered_sgld
from seqjax.inference.sgld import SGLDConfig


def test_buffered_sgld_runs() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    params = ARParameters()
    prior = HalfCauchyStds()
    _, obs, _, _ = simulate.simulate(key, target, None, params, sequence_length=5)

    pf = BootstrapParticleFilter(target, num_particles=4)
    config = BufferedSGLDConfig(
        buffer_size=1,
        batch_size=2,
        particle_filter=pf,
        parameter_prior=prior,
    )
    sgld_config = SGLDConfig(step_size=0.1, num_iters=3)
    samples = run_buffered_sgld(
        target, jrandom.PRNGKey(1), params, obs, config=config, sgld_config=sgld_config
    )

    assert samples.ar.shape == (sgld_config.num_iters,)

def test_buffered_sgld_step_tree() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    params = ARParameters()
    prior = HalfCauchyStds()
    _, obs, _, _ = simulate.simulate(key, target, None, params, sequence_length=5)

    pf = BootstrapParticleFilter(target, num_particles=4)
    step_sizes = ARParameters(ar=0.1, observation_std=0.0, transition_std=0.0)
    config = BufferedSGLDConfig(
        buffer_size=1,
        batch_size=2,
        particle_filter=pf,
        parameter_prior=prior,
    )
    sgld_config = SGLDConfig(step_size=step_sizes, num_iters=3)
    samples = run_buffered_sgld(
        target, jrandom.PRNGKey(1), params, obs, config=config, sgld_config=sgld_config
    )

    assert (samples.observation_std == params.observation_std).all()
