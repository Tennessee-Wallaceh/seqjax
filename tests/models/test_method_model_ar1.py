"""Tests for method-based sequential model API using AR(1)."""

import pytest
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model import evaluate, simulate
from seqjax.model.base import ObservationContext
from seqjax.model.typing import NoCondition


def test_ar1_method_model_simulate_and_log_prob_x():
    model = AR1Target()
    params = ARParameters()
    key = jrandom.PRNGKey(0)
    sequence_length = 8

    latents, observations = simulate.simulate_methods(
        key,
        model,
        parameters=params,
        sequence_length=sequence_length,
        condition=NoCondition(),
    )

    assert latents.batch_shape[0] == sequence_length + model.prior_order - 1
    assert observations.batch_shape[0] == sequence_length

    log_p_x = evaluate.log_prob_x_methods(
        model,
        latents,
        condition=NoCondition(),
        parameters=params,
    )
    assert jnp.isfinite(log_p_x)


def test_ar1_method_model_log_prob_matches_dispatch_api():
    method_model = AR1Target()
    params = ARParameters()
    key = jrandom.PRNGKey(1)
    sequence_length = 6

    latents, _ = simulate.simulate_methods(
        key,
        method_model,
        parameters=params,
        sequence_length=sequence_length,
        condition=NoCondition(),
    )

    log_p_method = evaluate.log_prob_x_methods(
        method_model,
        latents,
        condition=NoCondition(),
        parameters=params,
    )
    log_p_dispatched = evaluate.log_prob_x(
        method_model,
        latents,
        condition=NoCondition(),
        parameters=params,
    )

    assert jnp.allclose(log_p_method, log_p_dispatched)



def test_observation_context_lag_only_indexing():
    context = ObservationContext((1, 2, 3), max_length=3)
    assert context.max_length == 3
    assert context.current_length == 3
    assert context[-1] == 3
    assert context[-2] == 2

    next_context = context.append(4)
    assert next_context.current_length == 3
    assert next_context.to_tuple() == (2, 3, 4)

    with pytest.raises(IndexError):
        _ = context[0]

    with pytest.raises(TypeError):
        _ = context[1:]  # type: ignore[index]



def test_ar1_method_model_log_prob_joint_decomposes():
    model = AR1Target()
    params = ARParameters()
    key = jrandom.PRNGKey(2)

    latents, observations = simulate.simulate_methods(
        key,
        model,
        parameters=params,
        sequence_length=7,
        condition=NoCondition(),
    )

    log_p_x = evaluate.log_prob_x_methods(
        model,
        latents,
        condition=NoCondition(),
        parameters=params,
    )
    log_p_y_given_x = evaluate.log_prob_y_given_x_methods(
        model,
        latents,
        observations,
        condition=NoCondition(),
        parameters=params,
    )
    log_p_joint = evaluate.log_prob_joint_methods(
        model,
        latents,
        observations,
        condition=NoCondition(),
        parameters=params,
    )

    assert jnp.allclose(log_p_joint, log_p_x + log_p_y_given_x)
