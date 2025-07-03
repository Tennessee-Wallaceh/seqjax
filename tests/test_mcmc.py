import jax.random as jrandom
import jax.numpy as jnp

from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model import simulate
from seqjax.inference.mcmc import NUTSConfig, run_nuts


def test_run_nuts_shape() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()
    latents, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )

    config = NUTSConfig(num_warmup=5, num_samples=3, step_size=0.1)
    sample_key = jrandom.PRNGKey(1)
    samples = run_nuts(
        target,
        sample_key,
        parameters,
        observations,
        initial_latents=latents,
        config=config,
    )

    assert samples.x.shape == (config.num_samples, latents.x.shape[0])


def test_run_nuts_recovers_latents() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    true_params = ARParameters(
        ar=jnp.array(0.7), observation_std=jnp.array(0.05), transition_std=jnp.array(0.05)
    )
    latents, observations, _, _ = simulate.simulate(
        key, target, None, true_params, sequence_length=3
    )

    config = NUTSConfig(num_warmup=10, num_samples=20, step_size=0.03)
    sample_key = jrandom.PRNGKey(1)
    samples = run_nuts(
        target,
        sample_key,
        true_params,
        observations,
        initial_latents=latents,
        config=config,
    )

    mean_latents = jnp.mean(samples.x, axis=0)
    assert jnp.allclose(mean_latents, latents.x, atol=0.1)
