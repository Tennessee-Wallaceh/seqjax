import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.vi.conv_nf import LocalParityCoupling, _affine_with_min_scale


def test_local_parity_coupling_coord_rank_map_matches_parity() -> None:
    even = LocalParityCoupling(
        key=jrandom.key(0),
        transformer=_affine_with_min_scale(1e-6),
        sequence_dim=6,
        target_dim=5,
        update_even=True,
        cond_dim=3,
        kernel_size=3,
        nn_width=16,
        nn_depth=1,
    )
    odd = LocalParityCoupling(
        key=jrandom.key(1),
        transformer=_affine_with_min_scale(1e-6),
        sequence_dim=6,
        target_dim=5,
        update_even=False,
        cond_dim=3,
        kernel_size=3,
        nn_width=16,
        nn_depth=1,
    )

    assert jnp.array_equal(even._coord_to_rank, jnp.arange(5, dtype=jnp.int32))
    assert jnp.array_equal(odd._coord_to_rank, jnp.arange(4, -1, -1, dtype=jnp.int32))


def test_local_parity_coupling_inverse_round_trip_for_reverse_parity() -> None:
    bijection = LocalParityCoupling(
        key=jrandom.key(0),
        transformer=_affine_with_min_scale(1e-6),
        sequence_dim=6,
        target_dim=4,
        update_even=False,
        cond_dim=4,
        kernel_size=3,
        nn_width=32,
        nn_depth=2,
    )
    x = jrandom.normal(jrandom.key(1), (6, 4))
    condition = jrandom.normal(jrandom.key(2), (6, 4))

    y, _ = bijection.transform_and_log_det(x, condition)
    x_recovered, _ = bijection.inverse_and_log_det(y, condition)

    max_abs_err = jnp.max(jnp.abs(x - x_recovered))
    assert float(max_abs_err) < 0.015
