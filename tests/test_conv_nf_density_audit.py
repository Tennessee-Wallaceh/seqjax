from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.vi.conv_nf import LocalParityCoupling, _affine_with_min_scale


def _jacobian_logabsdet(
    bijection: LocalParityCoupling,
    x: jax.Array,
    condition: jax.Array,
) -> jax.Array:
    sequence_dim, target_dim = x.shape

    def flatten_transform(flat_x: jax.Array) -> jax.Array:
        transformed, _ = bijection.transform_and_log_det(
            flat_x.reshape(sequence_dim, target_dim),
            condition,
        )
        return transformed.reshape(-1)

    jacobian = jax.jacrev(flatten_transform)(x.reshape(-1))
    _, logabsdet = jnp.linalg.slogdet(jacobian)
    return logabsdet


def test_local_parity_coupling_log_det_matches_jacobian_for_reverse_rank_parity() -> None:
    bijection = LocalParityCoupling(
        key=jrandom.key(0),
        transformer=_affine_with_min_scale(1e-6),
        sequence_dim=2,
        target_dim=2,
        update_even=False,
        cond_dim=3,
        kernel_size=3,
        nn_width=16,
        nn_depth=1,
    )
    x = jrandom.normal(jrandom.key(1), (2, 2))
    condition = jrandom.normal(jrandom.key(2), (2, 3))

    _, reported_log_det = bijection.transform_and_log_det(x, condition)
    jacobian_log_det = _jacobian_logabsdet(bijection, x, condition)

    assert jnp.allclose(reported_log_det, jacobian_log_det, atol=1e-5)


def test_local_parity_coupling_log_det_matches_jacobian_for_forward_rank_parity() -> None:
    bijection = LocalParityCoupling(
        key=jrandom.key(0),
        transformer=_affine_with_min_scale(1e-6),
        sequence_dim=2,
        target_dim=2,
        update_even=True,
        cond_dim=3,
        kernel_size=3,
        nn_width=16,
        nn_depth=1,
    )
    x = jrandom.normal(jrandom.key(1), (2, 2))
    condition = jrandom.normal(jrandom.key(2), (2, 3))

    _, reported_log_det = bijection.transform_and_log_det(x, condition)
    jacobian_log_det = _jacobian_logabsdet(bijection, x, condition)

    assert jnp.allclose(reported_log_det, jacobian_log_det, atol=1e-5)
