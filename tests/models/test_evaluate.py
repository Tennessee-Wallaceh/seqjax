"""Evaluation tests for the method-based model API."""

import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import evaluate, simulate
from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model.double_well import DoubleWellTarget, DoubleWellParams, make_unit_time_increments
from seqjax.model.stochastic_vol import (
    SimpleStochasticVol,
    LogVolRW,
    make_constant_time_increments,
)
from seqjax.model.typing import NoCondition


def test_log_prob_joint_decomposes_ar1():
    target = AR1Target()
    params = ARParameters()
    key = jrandom.PRNGKey(0)

    x_path, y_path = simulate.simulate(
        key,
        target,
        params,
        sequence_length=8,
        condition=NoCondition(),
    )

    log_p_x = evaluate.log_prob_x(target, x_path, NoCondition(), params)
    log_p_y_given_x = evaluate.log_prob_y_given_x(
        target,
        x_path,
        y_path,
        NoCondition(),
        params,
    )
    log_joint = evaluate.log_prob_joint(
        target,
        x_path,
        y_path,
        NoCondition(),
        params,
    )

    assert jnp.allclose(log_joint, log_p_x + log_p_y_given_x)


def test_log_prob_finite_double_well():
    target = DoubleWellTarget()
    params = DoubleWellParams()
    key = jrandom.PRNGKey(1)
    condition = make_unit_time_increments(6, dt=0.1)

    x_path, y_path = simulate.simulate(
        key,
        target,
        params,
        sequence_length=6,
        condition=condition,
    )

    assert jnp.isfinite(evaluate.log_prob_x(target, x_path, condition, params))
    assert jnp.isfinite(
        evaluate.log_prob_y_given_x(target, x_path, y_path, condition, params)
    )


def test_log_prob_finite_simple_stochastic_vol():
    target = SimpleStochasticVol()
    params = LogVolRW(
        std_log_vol=jnp.array(3.2),
        mean_reversion=jnp.array(12.0),
        long_term_vol=jnp.array(0.16),
    )
    key = jrandom.PRNGKey(2)
    condition = make_constant_time_increments(5, dt=1.0 / (256 * 8), prior_order=1)

    x_path, y_path = simulate.simulate(
        key,
        target,
        params,
        sequence_length=5,
        condition=condition,
    )

    assert jnp.isfinite(evaluate.log_prob_joint(target, x_path, y_path, condition, params))
