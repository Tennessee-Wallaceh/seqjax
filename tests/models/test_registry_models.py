"""Tests covering all models registered in ``seqjax.model.registry``."""

from __future__ import annotations

import pytest
import jax.numpy as jnp
import jax.random as jrandom

from seqjax import util
from seqjax.model import evaluate, registry, simulate


def _iter_registry_entries():
    """Yield parameterised test cases for every registered model/preset."""

    for label, model_cls in registry.sequential_models.items():
        presets = registry.parameter_settings[label]
        for preset_name, parameters in presets.items():
            yield pytest.param(
                label,
                preset_name,
                model_cls,
                parameters,
                id=f"{label}[{preset_name}]",
            )


@pytest.mark.parametrize(
    "model_label, parameter_label, target_cls, parameters",
    list(_iter_registry_entries()),
)
def test_registered_models_can_simulate_and_evaluate(
    model_label: registry.SequentialModelLabel,
    parameter_label: str,
    target_cls,
    parameters,
):
    """Every registered preset should simulate and evaluate without errors."""

    case_id = f"{model_label}[{parameter_label}]"
    target = target_cls()
    sequence_length = 5
    key = jrandom.PRNGKey(0)

    condition_factory = registry.condition_generators.get(model_label)
    condition = (
        None
        if condition_factory is None
        else condition_factory(sequence_length)
    )

    latents, observations, latent_history, observation_history = simulate.simulate(
        key,
        target,
        condition=condition,
        parameters=parameters,
        sequence_length=sequence_length,
    )

    assert latents.batch_shape[0] == sequence_length, case_id
    assert observations.batch_shape[0] == sequence_length, case_id

    observation_path = util.concat_pytree(observation_history, observations)

    log_p_x = evaluate.log_prob_x(
        target,
        latents,
        condition=condition,
        parameters=parameters,
        x_history=latent_history,
    )
    log_p_y_given_x = evaluate.log_prob_y_given_x(
        target,
        latents,
        observation_path,
        condition=condition,
        parameters=parameters,
        x_history=latent_history,
    )
    log_p_joint = evaluate.log_prob_joint(
        target,
        latents,
        observation_path,
        condition=condition,
        parameters=parameters,
        x_history=latent_history,
    )

    assert jnp.isfinite(log_p_x), case_id
    assert jnp.isfinite(log_p_y_given_x), case_id
    assert jnp.isfinite(log_p_joint), case_id
    assert log_p_joint == pytest.approx(log_p_x + log_p_y_given_x), case_id
