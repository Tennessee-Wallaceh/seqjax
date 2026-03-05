import jax
import pytest

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def _build_target_posterior():
    generative_parameters = model_registry.parameter_settings["ar"]["base"]
    return model_registry.posterior_factories["ar"](generative_parameters)


def _make_inputs(target_posterior, sample_length: int):
    observations = target_posterior.target.observation_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.observation_cls.flat_dim))
    )
    conditions = target_posterior.target.condition_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.condition_cls.flat_dim))
    )
    parameters = target_posterior.inference_parameter_cls.unravel(
        jax.numpy.ones((target_posterior.inference_parameter_cls.flat_dim,))
    )
    return observations, conditions, parameters


def test_passthrough_embedder_supports_sample_positional_augmentation() -> None:
    sample_length = 6
    target_posterior = _build_target_posterior()

    embedding = vi.registry._build_embedder(
        vi.registry.PassthroughEmbedder(position_mode="sample", n_pos_embedding=2),
        target_posterior=target_posterior,
        sequence_length=9,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    observations, conditions, parameters = _make_inputs(target_posterior, sample_length)
    context = embedding.embed(observations, conditions, parameters)

    assert context.sequence_embedded_context.shape == (sample_length, 1 + 5)


def test_window_embedder_sequence_positional_requires_sequence_start() -> None:
    sample_length = 6
    target_posterior = _build_target_posterior()

    embedding = vi.registry._build_embedder(
        vi.registry.ShortContextEmbedder(position_mode="sequence", n_pos_embedding=2),
        target_posterior=target_posterior,
        sequence_length=12,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )
    observations, conditions, parameters = _make_inputs(target_posterior, sample_length)

    with pytest.raises(ValueError, match="sequence_start must be provided"):
        embedding.embed(observations, conditions, parameters)


def test_window_embedder_sequence_positional_adds_global_position_channel() -> None:
    sample_length = 6
    target_posterior = _build_target_posterior()

    embedding = vi.registry._build_embedder(
        vi.registry.ShortContextEmbedder(position_mode="sequence", n_pos_embedding=2),
        target_posterior=target_posterior,
        sequence_length=12,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )
    observations, conditions, parameters = _make_inputs(target_posterior, sample_length)

    context = embedding.embed(observations, conditions, parameters, sequence_start=3)

    # short-window default is prev/post 2 with y_dim=1 => 5 base dims + 5 positional dims
    assert context.sequence_embedded_context.shape == (sample_length, 10)
    expected_positions = (jax.numpy.arange(sample_length) + 3.5) / 12
    assert jax.numpy.allclose(context.sequence_embedded_context[:, 5], expected_positions)
