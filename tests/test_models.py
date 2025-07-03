import pytest

# mark requires jax
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from seqjax.model import simulate, evaluate
from seqjax.util import pytree_shape
from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model.stochastic_vol import SimpleStochasticVol, LogVolRW, TimeIncrement


def test_ar1_target_simulate_and_logp() -> None:
    key = jax.random.PRNGKey(0)
    params = ARParameters()
    latent, obs = simulate.simulate(key, AR1Target, None, params, sequence_length=3)

    assert latent.x.shape == (3,)
    assert obs.y.shape == (3,)

    logp = evaluate.log_p_joint(AR1Target, latent, obs, None, params)
    assert jnp.shape(logp) == ()


def test_simple_stochastic_vol_simulate_and_logp() -> None:
    key = jax.random.PRNGKey(0)
    params = LogVolRW(
        std_log_vol=jnp.array(0.1),
        mean_reversion=jnp.array(0.1),
        long_term_vol=jnp.array(1.0),
    )
    cond = TimeIncrement(jnp.array([1.0, 1.0, 1.0, 1.0]))
    latent, obs = simulate.simulate(
        key, SimpleStochasticVol, cond, params, sequence_length=3
    )

    assert latent.log_vol.shape == (4,)
    assert obs.underlying.shape == (4,)

    logp = evaluate.log_p_joint(SimpleStochasticVol, latent, obs, cond, params)
    assert jnp.shape(logp) == ()


@pytest.mark.parametrize(
    "target,params,condition,sequence_length,expect_lat,expect_obs,should_fail",
    [
        (
            AR1Target,
            ARParameters(),
            None,
            3,
            3 + AR1Target.prior.order - 1,
            3 + AR1Target.emission.observation_dependency,
            False,
        ),
        (
            SimpleStochasticVol,
            LogVolRW(
                std_log_vol=jnp.array(0.1),
                mean_reversion=jnp.array(0.1),
                long_term_vol=jnp.array(1.0),
            ),
            TimeIncrement(jnp.ones(4)),
            3,
            3 + SimpleStochasticVol.prior.order - 1,
            3 + SimpleStochasticVol.emission.observation_dependency,
            False,
        ),
        (
            SimpleStochasticVol,
            LogVolRW(
                std_log_vol=jnp.array(0.1),
                mean_reversion=jnp.array(0.1),
                long_term_vol=jnp.array(1.0),
            ),
            TimeIncrement(jnp.ones(3)),  # too short
            3,
            None,
            None,
            True,
        ),
    ],
)
def test_simulate_dependency_lengths(
    target,
    params,
    condition,
    sequence_length,
    expect_lat,
    expect_obs,
    should_fail,
) -> None:
    key = jax.random.PRNGKey(1)
    if should_fail:
        with pytest.raises(jax.errors.JaxRuntimeError):
            simulate.simulate(key, target, condition, params, sequence_length)
    else:
        latent, obs = simulate.simulate(key, target, condition, params, sequence_length)
        assert pytree_shape(latent)[0][0] == expect_lat
        assert pytree_shape(obs)[0][0] == expect_obs
