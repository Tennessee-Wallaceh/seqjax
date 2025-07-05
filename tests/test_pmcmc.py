import jax.random as jrandom
import jax.numpy as jnp

from functools import partial

from seqjax.inference.pmcmc import (
    ParticleMCMCConfig,
    run_particle_mcmc,
)
from seqjax.inference.mcmc import RandomWalkConfig, run_random_walk_metropolis, NUTSConfig, run_nuts_parameters
from seqjax.inference.particlefilter import BootstrapParticleFilter
from seqjax.model.ar import AR1Target, ARParameters, HalfCauchyStds
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
    mcmc = partial(run_random_walk_metropolis, config=rw_cfg)
    config = ParticleMCMCConfig(
        mcmc=mcmc,
        particle_filter=pf,
    )
    sample_key = jrandom.PRNGKey(1)
    samples = run_particle_mcmc(
        target,
        sample_key,
        observations,
        parameter_prior=HalfCauchyStds(),
        config=config,
        initial_parameters=parameters,
        initial_conditions=(None,),
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
    mcmc = partial(run_random_walk_metropolis, config=rw_cfg)
    config = ParticleMCMCConfig(
        mcmc=mcmc,
        particle_filter=pf,
    )
    sample_key = jrandom.PRNGKey(1)
    samples = run_particle_mcmc(
        target,
        sample_key,
        observations,
        parameter_prior=HalfCauchyStds(),
        config=config,
        initial_parameters=true_params,
        initial_conditions=(None,),
    )

    mean_ar = jnp.mean(samples.ar)
    assert jnp.allclose(mean_ar, true_params.ar, atol=0.1)


def test_particle_mcmc_with_nuts() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()
    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )

    pf = BootstrapParticleFilter(target, num_particles=5)
    nuts_cfg = NUTSConfig(num_warmup=5, num_samples=3, step_size=0.1)
    mcmc = partial(run_nuts_parameters, config=nuts_cfg)
    config = ParticleMCMCConfig(
        mcmc=mcmc,
        particle_filter=pf,
    )
    sample_key = jrandom.PRNGKey(1)
    samples = run_particle_mcmc(
        target,
        sample_key,
        observations,
        parameter_prior=HalfCauchyStds(),
        config=config,
        initial_parameters=parameters,
        initial_conditions=(None,),
    )

    assert samples.ar.shape == (nuts_cfg.num_samples,)

