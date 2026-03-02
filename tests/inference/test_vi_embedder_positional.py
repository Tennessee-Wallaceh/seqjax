import jax
import pytest

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def _build_target_posterior():
    generative_parameters = model_registry.parameter_settings["ar"]["base"]
    return model_registry.posterior_factories["ar"](generative_parameters)


def test_positional_embedder_builds_and_embeds_sequence_context() -> None:
    sample_length = 6
    sequence_length = 8
    n_pos_embedding = 4

    target_posterior = _build_target_posterior()

    positional = vi.registry.PositionalEmbedderConfig(n_pos_embedding=n_pos_embedding)

    embedding = vi.registry._build_embedder(
        positional,
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

    assert context.sequence_embedded_context.shape == (sample_length, 1 + 2 * n_pos_embedding)
    assert context.embedded_context.shape == observations.ravel().shape


def test_positional_embedder_rejects_invalid_n_pos_embedding() -> None:
    target_posterior = _build_target_posterior()

    with pytest.raises(ValueError, match="n_pos_embedding must be >= 1"):
        vi.embedder.PositionalEmbedder(
            target_posterior=target_posterior,
            sample_length=4,
            sequence_length=4,
            n_pos_embedding=0,
        )
