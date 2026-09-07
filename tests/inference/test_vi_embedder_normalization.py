import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def _build_double_well_posterior():
    hyperparameters = model_registry.hyperparameter_settings["double-well"]["base"]
    return model_registry.posterior_factories["double-well-ebonly"](hyperparameters)


def _build_stateful_embedder(config, target_posterior, sample_length: int):
    @eqx.nn.make_with_state
    def build():
        return vi.embedder.build_embedder(
            config,
            target=target_posterior.target,
            inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
            sequence_length=sample_length,
            sample_length=sample_length,
            embedding_key=jax.random.PRNGKey(0),
        )

    return build()


def _inputs(target_posterior, sample_length: int):
    observations = target_posterior.target.observation_cls.unravel(
        jnp.arange(sample_length, dtype=jnp.float32).reshape(sample_length, 1)
    )
    conditions = target_posterior.target.condition_cls.unravel(
        jnp.arange(1, sample_length + 1, dtype=jnp.float32).reshape(sample_length, 1)
    )
    parameter_dim = target_posterior.parameterization.inference_parameter_cls.flat_dim
    parameters = target_posterior.parameterization.inference_parameter_cls.unravel(
        jnp.arange(1, parameter_dim + 1, dtype=jnp.float32)
    )
    return observations, conditions, parameters


def test_null_normalization_preserves_existing_passthrough_features() -> None:
    sample_length = 5
    target_posterior = _build_double_well_posterior()
    embedding = vi.embedder.build_embedder(
        vi.embedder.PassthroughEmbedder(),
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sample_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )
    observations, conditions, parameters = _inputs(target_posterior, sample_length)

    context, state = embedding.embed(observations, conditions, parameters)

    assert state is None
    assert jnp.array_equal(context.sequence_features, observations.ravel())
    assert jnp.array_equal(context.flat_features, observations.ravel().flatten())
    assert (
        context.sequence_features.shape[-1]
        == target_posterior.target.observation_cls.flat_dim
    )


def test_normalized_conditions_and_parameters_are_added_to_feature_views() -> None:
    sample_length = 5
    target_posterior = _build_double_well_posterior()
    config = vi.embedder.PassthroughEmbedder(
        normalization=vi.embedder.NormalizationConfig(
            observation="conditional-ema",
            condition="ema",
            parameter="ema",
        )
    )
    embedding, state = _build_stateful_embedder(config, target_posterior, sample_length)
    observations, conditions, parameters = _inputs(target_posterior, sample_length)

    context, state = embedding.embed(
        observations, conditions, parameters, state, training=True
    )

    observation_dim = target_posterior.target.observation_cls.flat_dim
    condition_dim = target_posterior.target.condition_cls.flat_dim
    parameter_dim = target_posterior.parameterization.inference_parameter_cls.flat_dim
    assert context.sequence_features.shape == (
        sample_length,
        observation_dim + condition_dim,
    )
    assert context.flat_features.shape == (
        sample_length * observation_dim + parameter_dim,
    )
    assert jnp.allclose(jnp.mean(context.sequence_features, axis=0), 0.0, atol=1e-5)
    assert jnp.array_equal(context.observation_context.ravel(), observations.ravel())
    assert jnp.array_equal(context.condition_context.ravel(), conditions.ravel())
    assert jnp.array_equal(context.parameter_context.ravel(), parameters.ravel())


def test_enabled_normalization_requires_state() -> None:
    sample_length = 3
    target_posterior = _build_double_well_posterior()
    embedding = vi.embedder.build_embedder(
        vi.embedder.PassthroughEmbedder(
            normalization=vi.embedder.NormalizationConfig(condition="ema")
        ),
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sample_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )
    observations, conditions, parameters = _inputs(target_posterior, sample_length)

    with pytest.raises(ValueError, match="normalization state is required"):
        embedding.embed(observations, conditions, parameters)
