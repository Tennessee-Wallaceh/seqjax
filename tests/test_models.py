import pytest

# mark requires jax
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from seqjax import simulate, evaluate
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
