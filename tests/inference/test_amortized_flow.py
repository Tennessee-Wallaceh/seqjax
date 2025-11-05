import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference import embedder
from seqjax.inference.vi import autoregressive, base
from seqjax.model import ar, registry as model_registry, simulate


def _make_ar1_problem(sequence_length: int = 6):
    parameters = model_registry.parameter_settings["ar"]["base"]
    posterior = ar.AR1Bayesian(parameters)
    _latents, observations, _latent_hist, _obs_hist = simulate.simulate(
        jrandom.PRNGKey(0),
        posterior.target,
        condition=None,
        parameters=parameters,
        sequence_length=sequence_length,
    )
    return posterior, observations


def test_amortized_flow_sample_shape() -> None:
    buffer_length = 1
    batch_length = 2
    sample_length = 2 * buffer_length + batch_length
    context_dim = 3
    parameter_dim = 2
    flow = base.AmortizedMaskedAutoregressiveFlow(
        ar.LatentValue,
        buffer_length=buffer_length,
        batch_length=batch_length,
        context_dim=context_dim,
        parameter_dim=parameter_dim,
        key=jrandom.PRNGKey(1),
        nn_width=4,
        nn_depth=1,
        conditioner_width=5,
        conditioner_depth=1,
        conditioner_out_dim=4,
    )
    theta_context = jnp.ones((sample_length, parameter_dim))
    observation_context = jnp.ones((sample_length, context_dim))
    sample, log_q = flow.sample_and_log_prob(
        jrandom.PRNGKey(2), (theta_context, observation_context)
    )

    assert sample.x.shape == (sample_length,)
    assert log_q.shape == ()


def test_amortized_flow_respects_conditioning() -> None:
    buffer_length = 1
    batch_length = 2
    sample_length = 2 * buffer_length + batch_length
    context_dim = 3
    parameter_dim = 2
    flow = base.AmortizedMaskedAutoregressiveFlow(
        ar.LatentValue,
        buffer_length=buffer_length,
        batch_length=batch_length,
        context_dim=context_dim,
        parameter_dim=parameter_dim,
        key=jrandom.PRNGKey(3),
        nn_width=4,
        nn_depth=1,
        conditioner_width=5,
        conditioner_depth=1,
        conditioner_out_dim=4,
    )
    theta_context = jnp.zeros((sample_length, parameter_dim))
    observation_context = jnp.zeros((sample_length, context_dim))

    base_key = jrandom.PRNGKey(4)
    sample_a, _ = flow.sample_and_log_prob(
        base_key, (theta_context, observation_context)
    )
    sample_b, _ = flow.sample_and_log_prob(
        base_key, (theta_context, observation_context + 1.0)
    )

    assert not jnp.allclose(sample_a.x, sample_b.x)


def test_buffered_elbo_handles_scalar_and_vector_log_q() -> None:
    posterior, observations = _make_ar1_problem(sequence_length=6)
    sequence_length = observations.batch_shape[0]
    embed = embedder.WindowEmbedder(
        sequence_length,
        prev_window=1,
        post_window=1,
        y_dimension=observations.flat_dim,
    )

    parameter_dim = posterior.inference_parameter_cls.flat_dim
    latent_class = posterior.target.latent_cls

    autoreg_latent = autoregressive.AmortizedUnivariateAutoregressor(
        latent_class,
        buffer_length=1,
        batch_length=2,
        context_dim=embed.context_dimension,
        parameter_dim=parameter_dim,
        lag_order=1,
        nn_width=4,
        nn_depth=1,
        key=jrandom.PRNGKey(5),
    )

    flow_latent = base.AmortizedMaskedAutoregressiveFlow(
        latent_class,
        buffer_length=1,
        batch_length=2,
        context_dim=embed.context_dimension,
        parameter_dim=parameter_dim,
        key=jrandom.PRNGKey(6),
        nn_width=4,
        nn_depth=1,
        conditioner_width=5,
        conditioner_depth=1,
        conditioner_out_dim=4,
    )

    param_autoreg = base.MeanField(posterior.inference_parameter_cls)
    param_flow = base.MeanField(posterior.inference_parameter_cls)

    approximations = [
        base.BufferedSSMVI(autoreg_latent, param_autoreg, embed),
        base.BufferedSSMVI(flow_latent, param_flow, embed),
    ]

    for ix, approximation in enumerate(approximations):
        key = jrandom.fold_in(jrandom.PRNGKey(7), ix)
        loss = approximation.estimate_loss(
            observations,
            None,
            key,
            context_samples=1,
            samples_per_context=1,
            target_posterior=posterior,
            hyperparameters=None,
        )
        pretrain_loss = approximation.estimate_pretrain_loss(
            observations,
            None,
            key,
            context_samples=1,
            samples_per_context=1,
            target_posterior=posterior,
            hyperparameters=None,
        )
        assert jnp.ndim(loss) == 0
        assert jnp.ndim(pretrain_loss) == 0
        assert jnp.isfinite(loss)
        assert jnp.isfinite(pretrain_loss)
