import jax.random as jrandom

from seqjax.model.ar import AR1Target, ARParameters, AR1Bayesian
from seqjax.model.typing import HyperParameters
from seqjax.model import simulate
from seqjax.inference.mcmc import NUTSConfig, run_bayesian_nuts


def test_run_bayesian_nuts_shape() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()
    latents_true, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )

    config = NUTSConfig(num_warmup=5, num_samples=3, step_size=0.1)
    sample_key = jrandom.PRNGKey(1)
    posterior = AR1Bayesian(parameters)
    time_array, latents, params, _ = run_bayesian_nuts(
        posterior,
        HyperParameters(),
        sample_key,
        observations,
        config=config,
    )

    assert latents.x.shape == (time_array.shape[0], latents_true.x.shape[0])
    assert params.ar.shape == (time_array.shape[0],)
