import jax.random as jrandom

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
        observations,
        parameters=parameters,
        initial_latents=latents,
        config=config,
    )

    assert samples.x.shape == (config.num_samples, latents.x.shape[0])
