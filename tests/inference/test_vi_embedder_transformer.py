import jax

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def test_transformer_embedder_builds_and_embeds_sequence_context() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    transformer = vi.embedder.TransformerEmbedderConfig(
        hidden_dim=16,
        depth=2,
        num_heads=2,
        mlp_multiplier=2,
        pool_dim=3,
    )

    embedding = vi.embedder.build_embedder(
        transformer,
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    observations = target_posterior.target.observation_cls.unravel(
        jax.numpy.ones(
            (sample_length, target_posterior.target.observation_cls.flat_dim)
        )
    )
    conditions = target_posterior.target.condition_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.condition_cls.flat_dim))
    )
    parameters = target_posterior.parameterization.inference_parameter_cls.unravel(
        jax.numpy.ones(
            (target_posterior.parameterization.inference_parameter_cls.flat_dim,)
        )
    )

    context, _ = embedding.embed(observations, conditions, parameters)

    assert context.sequence_features.shape == (sample_length, transformer.hidden_dim)
    assert context.flat_features.shape == (
        transformer.hidden_dim * transformer.pool_dim,
    )


def test_transformer_embedder_supports_sequence_positional_augmentation() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    transformer = vi.embedder.TransformerEmbedderConfig(
        hidden_dim=16,
        depth=2,
        num_heads=2,
        mlp_multiplier=2,
        pool_dim=3,
        position_mode="sequence",
        n_pos_embedding=2,
    )

    embedding = vi.embedder.build_embedder(
        transformer,
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    observations = target_posterior.target.observation_cls.unravel(
        jax.numpy.ones(
            (sample_length, target_posterior.target.observation_cls.flat_dim)
        )
    )
    conditions = target_posterior.target.condition_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.condition_cls.flat_dim))
    )
    parameters = target_posterior.parameterization.inference_parameter_cls.unravel(
        jax.numpy.ones(
            (target_posterior.parameterization.inference_parameter_cls.flat_dim,)
        )
    )

    context, _ = embedding.embed(observations, conditions, parameters, sequence_start=1)

    assert context.sequence_features.shape == (
        sample_length,
        transformer.hidden_dim + 5,
    )
    assert context.flat_features.shape == (
        (transformer.hidden_dim + 5) * transformer.pool_dim,
    )
