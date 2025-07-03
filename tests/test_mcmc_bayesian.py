import jax.random as jrandom

from seqjax.model.ar import AR1Target, ARParameters, HalfCauchyStds
from seqjax.model import simulate
from seqjax.inference.mcmc import NUTSConfig, run_bayesian_nuts


def test_run_bayesian_nuts_shape() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()
    latents, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )

    config = NUTSConfig(num_warmup=5, num_samples=3, step_size=0.1)
    sample_key = jrandom.PRNGKey(1)
    samples_latents, samples_params = run_bayesian_nuts(
        target,
        sample_key,
        observations,
        parameter_prior=HalfCauchyStds(),
        initial_latents=latents,
        initial_parameters=parameters,
        config=config,
    )

    assert samples_latents.x.shape == (config.num_samples, latents.x.shape[0])
    assert samples_params.ar.shape == (config.num_samples,)
