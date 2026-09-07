import jax
import pytest

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def _build_target_posterior():
    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    return model_registry.posterior_factories["ar-full"](generative_parameters)


def _make_dummy_inputs(target_posterior, sample_length: int):
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
    return observations, conditions, parameters


def test_positional_embedder_builds_and_embeds_sequence_context() -> None:
    sample_length = 6
    sequence_length = 8
    n_pos_embedding = 4

    target_posterior = _build_target_posterior()

    positional = vi.embedder.PositionalEmbedderConfig(n_pos_embedding=n_pos_embedding)

    embedding = vi.embedder.build_embedder(
        positional,
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    observations, conditions, parameters = _make_dummy_inputs(
        target_posterior, sample_length
    )

    context, _ = embedding.embed(observations, conditions, parameters)

    assert context.sequence_features.shape == (sample_length, 1 + 2 * n_pos_embedding)
    assert context.flat_features.shape == observations.ravel().flatten().shape


def test_positional_embedder_sequence_mode_requires_sequence_start() -> None:
    target_posterior = _build_target_posterior()
    sample_length = 5

    embedding = vi.embedder.build_embedder(
        vi.embedder.PositionalEmbedderConfig(position_mode="sequence"),
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=11,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(4),
    )
    observations, conditions, parameters = _make_dummy_inputs(
        target_posterior, sample_length
    )

    with pytest.raises(ValueError, match="sequence_start must be provided"):
        embedding.embed(observations, conditions, parameters)


def test_positional_embedder_sequence_mode_uses_global_sequence_position() -> None:
    target_posterior = _build_target_posterior()
    sample_length = 5
    sequence_length = 11
    n_pos_embedding = 3

    embedding = vi.embedder.build_embedder(
        vi.embedder.PositionalEmbedderConfig(
            n_pos_embedding=n_pos_embedding,
            position_mode="sequence",
        ),
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(5),
    )
    observations, conditions, parameters = _make_dummy_inputs(
        target_posterior, sample_length
    )

    context, _ = embedding.embed(
        observations,
        conditions,
        parameters,
        sequence_start=2,
    )

    expected_positions = (jax.numpy.arange(sample_length) + 2.5) / sequence_length
    assert jax.numpy.allclose(context.sequence_features[:, 0], expected_positions)


def test_positional_embedder_rejects_invalid_n_pos_embedding() -> None:
    target_posterior = _build_target_posterior()

    with pytest.raises(ValueError, match="n_pos_embedding must be >= 1"):
        vi.embedder.PositionalEmbedder(
            target=target_posterior.target,
            parameter_cls=target_posterior.parameterization.inference_parameter_cls,
            sample_length=4,
            sequence_length=4,
            n_pos_embedding=0,
        )
