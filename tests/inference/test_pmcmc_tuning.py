import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.particlefilter.filter_definitions import BootstrapParticleFilter
from seqjax.inference.pmcmc.pmmh import (
    ParticleMCMCConfig,
    _make_log_joint_estimator,
    run_particle_mcmc,
)
from seqjax.inference.pmcmc.tuning import (
    ParticleFilterTuningConfig,
    tune_particle_filter_variance,
)
from seqjax.model.ar import AR1Bayesian, ARParameters
from seqjax.model.simulate import simulate
from seqjax.model.typing import HyperParameters


def _build_test_components(sequence_length: int = 6, num_particles: int = 8):
    key = jrandom.PRNGKey(0)
    ref_params = ARParameters(
        ar=jnp.array(0.5),
        observation_std=jnp.array(0.3),
        transition_std=jnp.array(0.2),
    )
    posterior = AR1Bayesian(ref_params)
    _, observations, _, _ = simulate(
        key,
        posterior.target,
        None,
        ref_params,
        sequence_length,
    )
    base_filter = BootstrapParticleFilter(posterior.target, num_particles=num_particles)
    return posterior, HyperParameters(), observations, base_filter


def test_tune_particle_filter_variance_stops_when_variance_small() -> None:
    posterior, hyperparameters, observations, base_filter = _build_test_components()
    estimator = _make_log_joint_estimator(
        posterior,
        hyperparameters,
        observations,
        None,
    )
    tuning_config = ParticleFilterTuningConfig(
        target_variance=1e6,
        max_particles=32,
        replications=3,
        diagnostic_samples=2,
    )
    key = jrandom.PRNGKey(1)
    tuned_filter, diagnostics = tune_particle_filter_variance(
        estimator,
        base_filter,
        posterior,
        hyperparameters,
        tuning_config,
        key,
    )

    assert tuned_filter.num_particles == base_filter.num_particles
    assert len(diagnostics["particle_counts"]) == 1
    final_variance = float(jnp.max(diagnostics["per_parameter_variance"][-1]))
    assert final_variance <= tuning_config.target_variance


def test_tune_particle_filter_variance_hits_particle_cap() -> None:
    posterior, hyperparameters, observations, base_filter = _build_test_components(
        num_particles=4
    )
    estimator = _make_log_joint_estimator(
        posterior,
        hyperparameters,
        observations,
        None,
    )
    tuning_config = ParticleFilterTuningConfig(
        target_variance=1e-6,
        max_particles=32,
        replications=4,
        diagnostic_samples=2,
    )
    key = jrandom.PRNGKey(2)
    tuned_filter, diagnostics = tune_particle_filter_variance(
        estimator,
        base_filter,
        posterior,
        hyperparameters,
        tuning_config,
        key,
    )

    assert tuned_filter.num_particles == tuning_config.max_particles
    assert int(diagnostics["particle_counts"][-1]) == tuning_config.max_particles
    assert len(diagnostics["particle_counts"]) > 1


def test_run_particle_mcmc_reports_tuning_diagnostics() -> None:
    posterior, hyperparameters, observations, base_filter = _build_test_components(
        num_particles=4
    )
    tuning_config = ParticleFilterTuningConfig(
        target_variance=1e-6,
        max_particles=32,
        replications=4,
        diagnostic_samples=2,
    )
    config = ParticleMCMCConfig(
        particle_filter=base_filter,
        initial_parameter_guesses=3,
        tuning=tuning_config,
    )
    key = jrandom.PRNGKey(3)
    samples, diagnostics = run_particle_mcmc(
        posterior,
        hyperparameters,
        key,
        observations,
        None,
        test_samples=4,
        config=config,
    )

    assert len(diagnostics) == 2
    tuning_log = diagnostics[1]
    assert int(tuning_log["particle_counts"][-1]) == tuning_config.max_particles
    assert tuning_log["log_marginal_samples"][-1].shape[0] == (
        tuning_config.diagnostic_samples or 1
    )

