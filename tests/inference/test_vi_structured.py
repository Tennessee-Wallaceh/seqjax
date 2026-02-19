import jax
import jax.numpy as jnp

from seqjax.inference.vi.api import LatentContext
from seqjax.inference.vi.structured import StructuredPrecisionGaussian
from seqjax.model.ar import LatentValue
from seqjax.model.typing import NoCondition


def test_structured_precision_gaussian_sample_shape_and_finite_log_prob() -> None:
    sample_length = 7
    approximation = StructuredPrecisionGaussian(
        LatentValue,
        batch_length=5,
        buffer_length=1,
        context_dim=4,
        parameter_dim=2,
        condition_dim=NoCondition.flat_dim,
        hidden_dim=8,
        depth=1,
        key=jax.random.PRNGKey(0),
    )

    context = LatentContext.build_from_sequence_context(
        sequence_embedded_context=jnp.ones((sample_length, 4)),
        observations=LatentValue.unravel(jnp.zeros((sample_length, 1))),
        conditions=NoCondition.unravel(jnp.zeros((sample_length, 0))),
        parameters=LatentValue.unravel(jnp.zeros((2, 1))),
    )

    sample, log_prob = approximation.sample_and_log_prob(jax.random.PRNGKey(1), context)

    assert sample.ravel().shape == (sample_length, 1)
    assert jnp.isfinite(log_prob)
