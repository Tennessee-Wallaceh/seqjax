"""Tests for :mod:`seqjax.util`."""

import jax
import jax.numpy as jnp
from functools import partial

from seqjax.util import concat_pytree, index_pytree, dynamic_slice_pytree, pytree_shape


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


def test_index_pytree_basic() -> None:
    """``index_pytree`` should index leaves along the first dimension."""

    tree = {
        "a": jnp.arange(6).reshape(3, 2),
        "b": jnp.arange(9).reshape(3, 3),
    }

    single = index_pytree(tree, 1)
    assert jnp.array_equal(single["a"], jnp.array([2, 3]))
    assert jnp.array_equal(single["b"], jnp.array([3, 4, 5]))

    nested = index_pytree(tree, (1, 0))
    assert jnp.array_equal(nested["a"], jnp.array(2))
    assert jnp.array_equal(nested["b"], jnp.array(3))


def test_concat_pytree_basic() -> None:
    """``concat_pytree`` should concatenate matching leaves."""

    tree1 = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}
    tree2 = {"a": jnp.array([5, 6]), "b": jnp.array([7, 8])}

    result = concat_pytree(tree1, tree2, axis=0)
    assert jnp.array_equal(result["a"], jnp.array([1, 2, 5, 6]))
    assert jnp.array_equal(result["b"], jnp.array([3, 4, 7, 8]))
