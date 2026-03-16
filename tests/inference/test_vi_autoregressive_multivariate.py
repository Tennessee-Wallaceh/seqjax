from collections import OrderedDict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
import pytest

import seqjax.model.typing as seqjtyping
from seqjax.inference.vi.autoregressive import AmortizedMultivariateAutoregressor


class _Latent2D(seqjtyping.Latent):
    x: jnp.ndarray
    _shape_template = OrderedDict(
        x=jax.ShapeDtypeStruct((2,), jnp.float32),
    )


class _Latent1D(seqjtyping.Latent):
    x: jnp.ndarray
    _shape_template = OrderedDict(
        x=jax.ShapeDtypeStruct((1,), jnp.float32),
    )


def _embedder_stub() -> SimpleNamespace:
    return SimpleNamespace(
        sequence_embedded_context_dim=3,
        parameter_context_dim=4,
        condition_context_dim=5,
    )


def test_multivariate_autoregressor_conditional_shape_and_finite_logq() -> None:
    lag_order = 2
    ar = AmortizedMultivariateAutoregressor(
        _Latent2D,
        sample_length=7,
        embedder=_embedder_stub(),
        lag_order=lag_order,
        nn_width=8,
        nn_depth=1,
        key=jrandom.PRNGKey(0),
    )

    sample, log_q = ar.conditional(
        jrandom.PRNGKey(1),
        prev_x=(jnp.zeros((2,)), jnp.ones((2,))),
        previous_available_flag=jnp.array([True, False]),
        theta_context=jnp.ones((4,)),
        context=jnp.ones((3,)),
        condition_context=jnp.ones((5,)),
    )

    assert sample.shape == (2,)
    assert jnp.shape(log_q) == ()
    assert jnp.isfinite(log_q)


def test_multivariate_autoregressor_rejects_univariate_latent() -> None:
    with pytest.raises(ValueError, match="flat_dim >= 2"):
        AmortizedMultivariateAutoregressor(
            _Latent1D,
            sample_length=7,
            embedder=_embedder_stub(),
            lag_order=2,
            nn_width=8,
            nn_depth=1,
            key=jrandom.PRNGKey(0),
        )


def test_multivariate_autoregressor_validates_cholesky_size() -> None:
    ar = AmortizedMultivariateAutoregressor(
        _Latent2D,
        sample_length=7,
        embedder=_embedder_stub(),
        lag_order=2,
        nn_width=8,
        nn_depth=1,
        key=jrandom.PRNGKey(0),
    )

    with pytest.raises(ValueError, match="Invalid Cholesky parameter count"):
        ar._build_cholesky(jnp.ones((2,)))


def test_multivariate_autoregressor_matches_gaussian_logpdf() -> None:
    ar = AmortizedMultivariateAutoregressor(
        _Latent2D,
        sample_length=7,
        embedder=_embedder_stub(),
        lag_order=2,
        nn_width=8,
        nn_depth=1,
        key=jrandom.PRNGKey(0),
    )

    key = jrandom.PRNGKey(4)
    prev_x = (jnp.zeros((2,)), jnp.ones((2,)))
    previous_available_flag = jnp.array([True, False])
    theta_context = jnp.arange(4, dtype=jnp.float32)
    context = jnp.arange(3, dtype=jnp.float32)
    condition_context = jnp.arange(5, dtype=jnp.float32)

    sample, log_q = ar.conditional(
        key,
        prev_x=prev_x,
        previous_available_flag=previous_available_flag,
        theta_context=theta_context,
        context=context,
        condition_context=condition_context,
    )

    inputs = jnp.concatenate(
        [
            *[jnp.ravel(x) for x in prev_x],
            previous_available_flag.astype(jnp.float32),
            theta_context,
            context,
            condition_context,
        ]
    )
    trans_params = ar.amortizer_mlp(inputs)
    loc = trans_params[:2]
    chol = ar._build_cholesky(trans_params[2:])
    cov = chol @ chol.T

    expected_log_q = jstats.multivariate_normal.logpdf(sample, loc, cov)
    assert jnp.allclose(log_q, expected_log_q, atol=1e-5)
