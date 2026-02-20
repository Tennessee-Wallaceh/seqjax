import jax

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def test_transformer_embedder_builds_and_embeds_sequence_context() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar"]["base"]
    target_posterior = model_registry.posterior_factories["ar"](generative_parameters)

    transformer = vi.registry.TransformerEmbedderConfig(
        hidden_dim=16,
        depth=2,
        num_heads=2,
        mlp_multiplier=2,
        pool_dim=3,
    )

    embedding = vi.registry._build_embedder(
        transformer,
        target_posterior=target_posterior,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    observations = target_posterior.target.observation_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.observation_cls.flat_dim))
    )
    conditions = target_posterior.target.condition_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.condition_cls.flat_dim))
    )
    parameters = target_posterior.inference_parameter_cls.unravel(
        jax.numpy.ones((target_posterior.inference_parameter_cls.flat_dim,))
    )

    context = embedding.embed(observations, conditions, parameters)

    assert context.sequence_embedded_context.shape == (sample_length, transformer.hidden_dim)
    assert context.embedded_context.shape == (transformer.hidden_dim * transformer.pool_dim,)
