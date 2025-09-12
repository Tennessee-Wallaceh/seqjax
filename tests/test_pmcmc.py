import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.pmcmc import ParticleMCMCConfig, run_particle_mcmc
from seqjax.inference.mcmc import RandomWalkConfig
from seqjax.inference.particlefilter import BootstrapParticleFilter
from seqjax.model.ar import AR1Target, ARParameters, AR1Bayesian
from seqjax.model.typing import HyperParameters
from seqjax import simulate


def test_run_particle_mcmc_shape() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()
    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )

    pf = BootstrapParticleFilter(target, num_particles=5)
    rw_cfg = RandomWalkConfig(step_size=0.1, num_samples=3)
    config = ParticleMCMCConfig(mcmc=rw_cfg, particle_filter=pf)
    sample_key = jrandom.PRNGKey(1)
    posterior = AR1Bayesian(parameters)

    time_array, _, samples, _ = run_particle_mcmc(
        posterior,
        HyperParameters(),
        sample_key,
        observations,
        None,
        config,
    )

    assert samples.ar.shape == (rw_cfg.num_samples,)


def test_run_particle_mcmc_recovers_params() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    true_params = ARParameters(
        ar=jnp.array(0.6),
        observation_std=jnp.array(0.05),
        transition_std=jnp.array(0.05),
    )
    _, observations, _, _ = simulate.simulate(
        key, target, None, true_params, sequence_length=4
    )

    pf = BootstrapParticleFilter(target, num_particles=10)
    rw_cfg = RandomWalkConfig(step_size=0.05, num_samples=20)
    config = ParticleMCMCConfig(mcmc=rw_cfg, particle_filter=pf)
    sample_key = jrandom.PRNGKey(1)
    posterior = AR1Bayesian(true_params)

    time_array, _, samples, _ = run_particle_mcmc(
        posterior,
        HyperParameters(),
        sample_key,
        observations,
        None,
        config,
    )

    mean_ar = jnp.mean(samples.ar)
    assert jnp.allclose(mean_ar, true_params.ar, atol=0.1)
