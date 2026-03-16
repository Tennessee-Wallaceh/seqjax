from collections import OrderedDict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jrandom
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
