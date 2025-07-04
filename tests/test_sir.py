import jax.random as jrandom
import jax.numpy as jnp
import jax.scipy.stats as jstats

from seqjax.model.sir import SIRState, SIRParameters, SIRTransition


def _truncated_poisson_logpmf(x, lam, max_x):
    cdf = jstats.poisson.cdf(max_x - 1, lam)
    log_tail = jnp.log1p(-cdf)
    log_p = jnp.where(x < max_x, jstats.poisson.logpmf(x, lam), log_tail)
    return jnp.where(x > max_x, -jnp.inf, log_p)


def test_transition_log_prob_truncation() -> None:
    params = SIRParameters(
        infection_rate=jnp.array(10.0),
        recovery_rate=jnp.array(10.0),
        population=jnp.array(3.0),
    )
    state = SIRState(s=jnp.array(2.0), i=jnp.array(1.0), r=jnp.array(0.0))
    next_state = SIRState(s=jnp.array(0.0), i=jnp.array(0.0), r=jnp.array(3.0))

    logp = SIRTransition.log_prob((state,), next_state, None, params)

    lam_inf = params.infection_rate * state.s * state.i / params.population
    i_temp = state.i + state.s
    lam_rec = params.recovery_rate * i_temp
    expected = _truncated_poisson_logpmf(state.s, lam_inf, state.s)
    expected += _truncated_poisson_logpmf(i_temp, lam_rec, i_temp)

    assert jnp.allclose(logp, expected)


def test_transition_sample_truncation() -> None:
    params = SIRParameters(
        infection_rate=jnp.array(1e6),
        recovery_rate=jnp.array(1e6),
        population=jnp.array(3.0),
    )
    state = SIRState(s=jnp.array(2.0), i=jnp.array(1.0), r=jnp.array(0.0))
    next_state = SIRTransition.sample(jrandom.PRNGKey(0), (state,), None, params)

    new_inf = state.s - next_state.s
    new_rec = next_state.r - state.r

    assert new_inf == state.s
    assert new_rec == state.i + new_inf
