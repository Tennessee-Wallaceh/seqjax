import jax.random as jrandom
import jax.numpy as jnp

from seqjax.inference.autoregressive_vi import (
    RandomAutoregressor,
    AmortizedUnivariateAutoregressor,
    AmortizedMultivariateIsotropicAutoregressor,
    AutoregressiveVIConfig,
    run_autoregressive_vi,
)
from seqjax.inference.embedder import PassThroughEmbedder
from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model import simulate


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


def test_run_autoregressive_vi_shape() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    params = ARParameters()
    _, observations, _, _ = simulate.simulate(
        key, target, None, params, sequence_length=5
    )

    embedder = PassThroughEmbedder(
        sample_length=5, prev_window=0, post_window=0, y_dimension=1
    )
    sampler = RandomAutoregressor(
        sample_length=5,
        x_dim=1,
        context_dim=embedder.context_dimension,
        parameter_dim=3,
        lag_order=1,
    )
    config = AutoregressiveVIConfig(sampler=sampler, embedder=embedder, num_samples=2)

    samples = run_autoregressive_vi(
        target,
        jrandom.PRNGKey(1),
        observations,
        parameters=params,
        config=config,
    )

    assert samples.x.shape == (config.num_samples, 5)


def test_run_autoregressive_vi_return_parameters() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    params = ARParameters()
    _, observations, _, _ = simulate.simulate(
        key, target, None, params, sequence_length=5
    )

    embedder = PassThroughEmbedder(
        sample_length=5, prev_window=0, post_window=0, y_dimension=1
    )
    sampler = RandomAutoregressor(
        sample_length=5,
        x_dim=1,
        context_dim=embedder.context_dimension,
        parameter_dim=1,
        lag_order=1,
    )
    config = AutoregressiveVIConfig(
        sampler=sampler,
        embedder=embedder,
        num_samples=3,
        return_parameters=True,
        parameter_std=0.1,
    )

    latents, theta_samples = run_autoregressive_vi(
        target,
        jrandom.PRNGKey(2),
        observations,
        parameters=params,
        config=config,
    )

    assert latents.x.shape == (config.num_samples, 5)
    assert theta_samples.ar.shape == (config.num_samples,)
