import jax
import pytest

from seqjax.inference import vi
from seqjax.model import registry as model_registry


def _make_context(embedder, target_posterior, sample_length: int):
    observations = target_posterior.target.observation_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.observation_cls.flat_dim))
    )
    conditions = target_posterior.target.condition_cls.unravel(
        jax.numpy.ones((sample_length, target_posterior.target.condition_cls.flat_dim))
    )
    parameters = target_posterior.inference_parameter_cls.unravel(
        jax.numpy.ones((target_posterior.inference_parameter_cls.flat_dim,))
    )
    return embedder.embed(observations, conditions, parameters)


@pytest.mark.parametrize("pool_kind", ["avg", "max"])
def test_conv1d_embedder_uses_aggregator_pooling(pool_kind: str) -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar"]["base"]
    target_posterior = model_registry.posterior_factories["ar"](generative_parameters)

    config = vi.registry.Conv1DEmbedderConfig(
        hidden_dim=4,
        depth=2,
        kernel_size=3,
        pool_dim=3,
        pool_kind=pool_kind,
    )

    embedding = vi.registry._build_embedder(
        config,
        target_posterior=target_posterior,
        sequence_length=sequence_length,
        sample_length=sample_length,
        embedding_key=jax.random.PRNGKey(0),
    )

    context = _make_context(embedding, target_posterior, sample_length)

    assert context.sequence_embedded_context.shape == (sample_length, config.hidden_dim)
    assert context.embedded_context.shape == (config.hidden_dim * config.pool_dim,)


def test_conv1d_embedder_rejects_unknown_pool_kind() -> None:
    sample_length = 6
    sequence_length = 8

    generative_parameters = model_registry.parameter_settings["ar"]["base"]
    target_posterior = model_registry.posterior_factories["ar"](generative_parameters)

    with pytest.raises(ValueError, match="pool_kind: median not supported"):
        vi.registry._build_embedder(
            vi.registry.Conv1DEmbedderConfig(pool_kind="median"),  # type: ignore[arg-type]
            target_posterior=target_posterior,
            sequence_length=sequence_length,
            sample_length=sample_length,
            embedding_key=jax.random.PRNGKey(0),
        )
