import jax.random as jrandom
import jax.numpy as jnp

from seqjax.inference.autoregressive_vi import (
    RandomAutoregressor,
    AmortizedUnivariateAutoregressor,
    AmortizedMultivariateIsotropicAutoregressor,
)


def test_random_autoregressor_sample_shape() -> None:
    ar = RandomAutoregressor(
        sample_length=5,
        x_dim=2,
        context_dim=1,
        parameter_dim=1,
        lag_order=1,
    )
    theta = jnp.ones((1,))
    context = jnp.ones((5, 1))
    x, log_q = ar.sample_single_path(jrandom.PRNGKey(0), theta, context)
    assert x.shape == (5, 2)
    assert log_q.shape == ()


def test_amortized_univariate_autoregressor_sample_shape() -> None:
    ar = AmortizedUnivariateAutoregressor(
        sample_length=5,
        context_dim=1,
        parameter_dim=1,
        lag_order=1,
        nn_width=4,
        nn_depth=2,
        key=jrandom.PRNGKey(0),
    )
    theta = jnp.ones((1,))
    context = jnp.ones((5, 1))
    x, log_q = ar.sample_single_path(jrandom.PRNGKey(1), theta, context)
    assert x.shape == (5, 1)
    assert log_q.shape == (5,)


def test_amortized_multivariate_isotropic_autoregressor_sample_shape() -> None:
    ar = AmortizedMultivariateIsotropicAutoregressor(
        sample_length=5,
        context_dim=1,
        parameter_dim=1,
        lag_order=1,
        nn_width=4,
        nn_depth=2,
        x_dim=3,
        key=jrandom.PRNGKey(0),
    )
    theta = jnp.ones((1,))
    context = jnp.ones((5, 1))
    x, log_q = ar.sample_single_path(jrandom.PRNGKey(2), theta, context)
    assert x.shape == (5, 3)
    assert log_q.shape == ()
