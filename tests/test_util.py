"""Tests for :mod:`seqjax.util`."""

import jax
import jax.numpy as jnp
from functools import partial

from seqjax.util import dynamic_slice_pytree, pytree_shape


def test_pytree_shape_dict() -> None:
    """Check that ``pytree_shape`` returns shape and leaf count for dicts."""

    tree = {"a": jnp.zeros((2, 3)), "b": jnp.ones((2, 3))}
    shape, leaf_count = pytree_shape(tree)
    assert shape == (2, 3)
    assert leaf_count == 2


def test_dynamic_slice_pytree_matches_lax() -> None:
    """``dynamic_slice_pytree`` should match ``jax.lax.dynamic_slice_in_dim``."""

    tree = {"a": jnp.arange(10), "b": jnp.arange(10) * 2}
    start_index = 2
    slice_size = 5
    sliced = dynamic_slice_pytree(tree, start_index, slice_size)
    expected = jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=start_index,
            slice_size=slice_size,
            axis=0,
        ),
        tree,
    )

    assert jax.tree_util.tree_all(  # type: ignore[call-arg]
        jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), sliced, expected)
    )
