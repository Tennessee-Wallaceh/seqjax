import jax
import pytest

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def _make_context(embedder, target_posterior, sample_length: int):
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
    return embedder.embed(observations, conditions, parameters)[0]


@pytest.mark.parametrize("pool_kind", ["avg", "max"])
def test_conv1d_embedder_uses_aggregator_pooling(pool_kind: str) -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    config = vi.embedder.Conv1DEmbedderConfig(
        hidden_dim=4,
        depth=2,
        kernel_size=3,
        pool_dim=3,
        pool_kind=pool_kind,
    )

    embedding = vi.embedder.build_embedder(
        config,
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    context = _make_context(embedding, target_posterior, sample_length)

    assert context.sequence_features.shape == (sample_length, config.hidden_dim)
    assert context.flat_features.shape == (config.hidden_dim * config.pool_dim,)


def test_conv1d_embedder_rejects_unknown_pool_kind() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    with pytest.raises(ValueError, match="pool_kind: median not supported"):
        vi.embedder.build_embedder(
            vi.embedder.Conv1DEmbedderConfig(pool_kind="median"),  # type: ignore[arg-type]
            target=target_posterior.target,
            inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
            sequence_length=sequence_length,
            sample_length=sample_length,
            embedding_key=jax.random.PRNGKey(0),
        )


def test_conv1d_embedder_supports_positional_augmentation() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    config = vi.embedder.Conv1DEmbedderConfig(
        hidden_dim=4,
        depth=1,
        kernel_size=3,
        pool_dim=3,
        pool_kind="avg",
        position_mode="sample",
        n_pos_embedding=2,
    )

    embedding = vi.embedder.build_embedder(
        config,
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    context = _make_context(embedding, target_posterior, sample_length)
    assert context.sequence_features.shape == (sample_length, config.hidden_dim + 5)
    assert context.flat_features.shape == ((config.hidden_dim + 5) * config.pool_dim,)


def test_conv1d_embedder_builds_optional_normalizers() -> None:
    sample_length = 6
    generative_parameters = model_registry.parameter_settings["ar-full"]["base"]
    target_posterior = model_registry.posterior_factories["ar-full"](
        generative_parameters
    )

    embedding = vi.embedder.build_embedder(
        vi.embedder.Conv1DEmbedderConfig(
            hidden_dim=4,
            embed_norm_kind="layer-norm",
            param_norm=True,
        ),
        target=target_posterior.target,
        inference_parameter_cls=target_posterior.parameterization.inference_parameter_cls,
        sequence_length=8,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    assert embedding.embedding_norm is not None
    assert embedding.param_norm is not None
