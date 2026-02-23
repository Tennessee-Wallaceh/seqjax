import jax
import pytest

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def _build_context(embedding, target_posterior, sample_length: int):
    observations = target_posterior.target.observation_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.observation_cls.flat_dim))
    )
    conditions = target_posterior.target.condition_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.condition_cls.flat_dim))
    )
    parameters = target_posterior.inference_parameter_cls.unravel(
        jax.numpy.ones((target_posterior.inference_parameter_cls.flat_dim,))
    )
    return embedding.embed(observations, conditions, parameters)


def test_birnn_embedder_observation_flatten_default() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar"]["base"]
    target_posterior = model_registry.posterior_factories["ar"](generative_parameters)

    embedding = vi.registry._build_embedder(
        vi.registry.BiRNNEmbedder(hidden_dim=4),
        target_posterior=target_posterior,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    context = _build_context(embedding, target_posterior, sample_length)

    assert context.sequence_embedded_context.shape == (sample_length, 8)
    assert context.embedded_context.size == (
        sample_length * target_posterior.target.observation_cls.flat_dim
    )


def test_birnn_embedder_sequence_flatten_aggregation() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar"]["base"]
    target_posterior = model_registry.posterior_factories["ar"](generative_parameters)

    embedding = vi.registry._build_embedder(
        vi.registry.BiRNNEmbedder(hidden_dim=4, aggregation_kind="sequence-flatten"),
        target_posterior=target_posterior,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    context = _build_context(embedding, target_posterior, sample_length)

    assert context.sequence_embedded_context.shape == (sample_length, 8)
    assert context.embedded_context.shape == (sample_length * 8,)


def test_birnn_embedder_rejects_none_aggregation() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar"]["base"]
    target_posterior = model_registry.posterior_factories["ar"](generative_parameters)

    with pytest.raises(ValueError, match="aggregation_kind='none' is not supported"):
        vi.registry._build_embedder(
            vi.registry.BiRNNEmbedder(hidden_dim=4, aggregation_kind="none"),  # type: ignore[arg-type]
            target_posterior=target_posterior,
            sequence_length=sequence_length,
            sample_length=sample_length,
            embedding_key=jax.random.PRNGKey(0),
        )
