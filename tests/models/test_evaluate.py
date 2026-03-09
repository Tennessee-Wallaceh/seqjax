import jax
import jax.numpy as jnp
from collections import OrderedDict
from functools import partial
from typing import ClassVar

import seqjax.model.typing as seqjtyping
from seqjax.model import evaluate
from seqjax.model import interface as model_interface


class DummyLatent(seqjtyping.Latent):
    x: jax.Array
    _shape_template: ClassVar = OrderedDict(
        x=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class DummyObs(seqjtyping.Observation):
    y: jax.Array
    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class DummyParam(seqjtyping.Parameters):
    theta: jax.Array
    _shape_template: ClassVar = OrderedDict(
        theta=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


prior_order = 2
transition_order = 2
emission_order = 2
observation_dependency = 1

latent_context = partial(model_interface.LatentContext, length=2)
observation_context = partial(model_interface.ObservationContext, length=1)
condition_context = partial(model_interface.ConditionContext, length=2)


def prior_sample(key, conditions, parameters):
    return latent_context((DummyLatent(x=jnp.array(-1.0)), DummyLatent(x=jnp.array(0.0))))


def prior_log_prob(latent, conditions, parameters):
    return latent[0].x + latent[-1].x + conditions[0].c + conditions[-1].c + parameters.theta


def transition_sample(key, latent_history, condition, parameters):
    return DummyLatent(x=latent_history[0].x + latent_history[-1].x + condition.c)


def transition_log_prob(latent_history, latent, condition, parameters):
    return latent.x + latent_history[0].x + latent_history[-1].x + condition.c + parameters.theta


def emission_sample(key, latent_history, observation_history, condition, parameters):
    return DummyObs(y=latent_history[-1].x + observation_history[0].y + condition.c)


def emission_log_prob(latent_history, observation, observation_history, condition, parameters):
    return (
        observation.y
        + latent_history[0].x
        + latent_history[-1].x
        + observation_history[0].y
        + condition.c
        + parameters.theta
    )


class DummyCondition(seqjtyping.Condition):
    c: jax.Array
    _shape_template: ClassVar = OrderedDict(
        c=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


class DummyModel:
    latent_cls = DummyLatent
    observation_cls = DummyObs
    parameter_cls = DummyParam
    condition_cls = DummyCondition

    prior_order = prior_order
    transition_order = transition_order
    emission_order = emission_order
    observation_dependency = observation_dependency

    latent_context = staticmethod(latent_context)
    observation_context = staticmethod(observation_context)
    condition_context = staticmethod(condition_context)

    prior_sample = staticmethod(prior_sample)
    prior_log_prob = staticmethod(prior_log_prob)
    transition_sample = staticmethod(transition_sample)
    transition_log_prob = staticmethod(transition_log_prob)
    emission_sample = staticmethod(emission_sample)
    emission_log_prob = staticmethod(emission_log_prob)


def test_log_prob_factorization() -> None:
    target = DummyModel()
    x_path = DummyLatent(x=jnp.array([-1.0, 0.0, 2.0, 3.0]))
    observation_path = DummyObs(y=jnp.array([10.0, 11.0, 12.0, 13.0]))
    condition = DummyCondition(c=jnp.array([1.0, 2.0, 3.0, 4.0]))
    params = DummyParam(theta=jnp.array(0.5))

    log_p_x = evaluate.log_prob_x(target, x_path, condition, params)
    log_p_y_given_x = evaluate.log_prob_y_given_x(
        target,
        x_path,
        observation_path,
        condition,
        params,
    )
    log_p_joint = evaluate.log_prob_joint(
        target,
        x_path,
        observation_path,
        condition,
        params,
    )

    assert jnp.isclose(log_p_joint, log_p_x + log_p_y_given_x)


def test_log_prob_x_no_condition() -> None:
    from seqjax.model import ar

    target = ar
    params = ar.ARParameters()
    key = jax.random.PRNGKey(0)
    from seqjax.model import simulate

    x_path, y_path = simulate.simulate(
        key,
        target,
        params,
        sequence_length=5,
        condition=seqjtyping.NoCondition(),
    )

    out_x = evaluate.log_prob_x(target, x_path, seqjtyping.NoCondition(), params)
    out_y = evaluate.log_prob_y_given_x(target, x_path, y_path, seqjtyping.NoCondition(), params)

    assert jnp.isfinite(out_x)
    assert jnp.isfinite(out_y)


def test_log_prob_errors_on_short_condition() -> None:
    target = DummyModel()
    x_path = DummyLatent(x=jnp.array([-1.0, 0.0, 2.0]))
    observation_path = DummyObs(y=jnp.array([1.0, 2.0, 3.0]))
    short_condition = DummyCondition(c=jnp.array([1.0]))
    params = DummyParam(theta=jnp.array(0.0))

    try:
        evaluate.log_prob_y_given_x(
            target,
            x_path,
            observation_path,
            short_condition,
            params,
        )
    except ValueError as error:
        assert "condition length" in str(error)
    else:
        raise AssertionError("Expected ValueError for short condition")
