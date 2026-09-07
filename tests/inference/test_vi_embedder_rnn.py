import jax
import pytest

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def _build_context(embedding, target_posterior, sample_length: int):
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
    return embedding.embed(observations, conditions, parameters)[0]


def test_birnn_embedder_observation_flatten_default() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    embedding = vi.embedder.build_embedder(
        vi.embedder.BiRNNEmbedder(hidden_dim=4),
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    context = _build_context(embedding, target_posterior, sample_length)

    assert context.sequence_features.shape == (sample_length, 8)
    assert context.flat_features.size == (
        sample_length * target_posterior.target.observation_cls.flat_dim
    )


def test_birnn_embedder_sequence_flatten_aggregation() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    embedding = vi.embedder.build_embedder(
        vi.embedder.BiRNNEmbedder(hidden_dim=4, aggregation_kind="sequence-flatten"),
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    context = _build_context(embedding, target_posterior, sample_length)

    assert context.sequence_features.shape == (sample_length, 8)
    assert context.flat_features.shape == (sample_length * 8,)


def test_birnn_embedder_rejects_none_aggregation() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    with pytest.raises(ValueError, match="aggregation_kind='none' is not supported"):
        vi.embedder.build_embedder(
            vi.embedder.BiRNNEmbedder(hidden_dim=4, aggregation_kind="none"),  # type: ignore[arg-type]
            target=target_posterior.target,
            inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
            sequence_length=sequence_length,
            sample_length=sample_length,
            embedding_key=jax.random.PRNGKey(0),
        )


def test_birnn_embedder_supports_positional_augmentation() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    embedding = vi.embedder.build_embedder(
        vi.embedder.BiRNNEmbedder(
            hidden_dim=4,
            aggregation_kind="sequence-flatten",
            position_mode="sample",
            n_pos_embedding=2,
        ),
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    context = _build_context(embedding, target_posterior, sample_length)

    assert context.sequence_features.shape == (sample_length, 13)
    assert context.flat_features.shape == (sample_length * 13,)
