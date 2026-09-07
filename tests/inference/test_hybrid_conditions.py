import jax
import jax.numpy as jnp

from seqjax.inference.interface import ObservationDataset
from seqjax.inference.particlefilter import registry as pf_registry
from seqjax.inference.particlefilter.base import run_filter
from seqjax.inference import vi
from seqjax.model.simulate import simulate
from seqjax.model.condition import layout_for
from seqjax.model.stochastic_vol import time_variable_var


def _conditioned_sequence(length: int = 8):
    posterior = time_variable_var.svar_full()
    model_parameters = time_variable_var.LogVarParams(
        std_log_var=jnp.array(0.4),
        mean_reversion_rate=jnp.array(10.0),
        long_term_log_var=2.0 * jnp.log(jnp.array(0.2)),
    )
    inference_parameters = posterior.parameterization.from_model_parameters(
        model_parameters
    )
    conditions = time_variable_var.TimeStepCondition(
        timestep=jnp.linspace(1e-4, 8e-4, length)
    )
    _, observations = simulate(
        jax.random.key(1),
        posterior.target,
        model_parameters,
        length,
        conditions,
    )
    particle_filter = pf_registry.build_filter(
        posterior,
        pf_registry.BootstrapFilterConfig(
            resample="multinomial",
            num_particles=16,
        ),
    )
    return posterior, particle_filter, inference_parameters, observations, conditions


def test_filter_records_the_condition_for_each_model_step() -> None:
    _, particle_filter, parameters, observations, conditions = _conditioned_sequence()

    log_weights, _, (transition_dt, emission_dt) = run_filter(
        jax.random.key(2),
        particle_filter,
        parameters,
        observations,
        condition_path=conditions,
        recorders=(
            lambda data: data.transition_condition.timestep,
            lambda data: data.emission_condition.timestep,
        ),
    )

    assert jnp.all(jnp.isfinite(log_weights))
    assert jnp.array_equal(emission_dt, conditions.timestep)
    # Step zero has no transition; its recorded value is the condition used by
    # the initial emission. Every actual transition is destination-aligned.
    assert jnp.array_equal(transition_dt, conditions.timestep)


def test_hybrid_score_estimator_handles_time_varying_conditions() -> None:
    posterior, particle_filter, parameters, observations, conditions = (
        _conditioned_sequence()
    )
    dataset = ObservationDataset.from_single_sequence(observations, conditions)

    score = vi.registry.hybrid.buffered_score_estimate(
        particle_filter,
        posterior,
        dataset,
        parameters,
        jax.random.key(3),
        num_sequence_minibatch=1,
        batch_length=2,
        buffer_length=1,
    )

    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(score))


def test_condition_layout_rejects_a_short_path() -> None:
    _, particle_filter, parameters, observations, conditions = _conditioned_sequence()
    short_conditions = time_variable_var.TimeStepCondition(
        timestep=conditions.timestep[:-1]
    )

    try:
        run_filter(
            jax.random.key(4),
            particle_filter,
            parameters,
            observations,
            condition_path=short_conditions,
        )
    except ValueError as error:
        assert "Condition path is too short" in str(error)
    else:
        raise AssertionError("Expected an informative error for a short condition path")


def test_condition_layout_prepares_counts_from_observations() -> None:
    posterior = time_variable_var.svar_full()
    conditions = time_variable_var.TimeStepCondition(timestep=jnp.array([1e-4]))

    prepared = layout_for(posterior.target).prepare(
        posterior.target,
        conditions,
        observation_count=1,
    )

    assert prepared.transitions.batch_shape == (0,)
    assert prepared.recurrent_emissions.batch_shape == (0,)
    assert prepared.emissions.batch_shape == (1,)
    assert prepared.initial_emission.timestep == conditions.timestep[0]


def test_condition_layout_rejects_zero_observations() -> None:
    posterior = time_variable_var.svar_full()
    conditions = time_variable_var.TimeStepCondition(timestep=jnp.array([1e-4]))

    try:
        layout_for(posterior.target).prepare(
            posterior.target,
            conditions,
            observation_count=0,
        )
    except ValueError as error:
        assert "at least one observation" in str(error)
    else:
        raise AssertionError("Expected an informative error for zero observations")
