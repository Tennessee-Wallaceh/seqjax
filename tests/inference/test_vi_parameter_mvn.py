import jax
import jax.numpy as jnp

from seqjax.inference.vi.base import MultivariateNormal
from seqjax.inference.vi.registry import (
    MultivariateNormalParameterApproximation,
    _build_parameter_approximation,
)
from seqjax.model.ar import AROnlyParameters


def test_mvn_parameter_approximation_sample_shape_and_finite_log_prob() -> None:
    approximation = MultivariateNormal(AROnlyParameters)

    sample, log_prob = approximation.sample_and_log_prob(jax.random.PRNGKey(0), None)

    assert sample.ravel().shape == (AROnlyParameters.flat_dim,)
    assert jnp.isfinite(log_prob)


def test_registry_builds_transformed_mvn_parameter_approximation() -> None:
    approximation = _build_parameter_approximation(
        AROnlyParameters,
        MultivariateNormalParameterApproximation(),
        key=jax.random.PRNGKey(1),
    )

    sample, log_prob = approximation.sample_and_log_prob(jax.random.PRNGKey(2), None)

    assert sample.ravel().shape == (AROnlyParameters.flat_dim,)
    assert jnp.isfinite(log_prob)
