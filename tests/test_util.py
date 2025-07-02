"""Tests for :mod:`seqjax.util`."""

import jax.numpy as jnp

from seqjax.util import pytree_shape


def test_pytree_shape_dict() -> None:
    """Check that ``pytree_shape`` returns shape and leaf count for dicts."""

    tree = {"a": jnp.zeros((2, 3)), "b": jnp.ones((2, 3))}
    shape, leaf_count = pytree_shape(tree)
    assert shape == (2, 3)
    assert leaf_count == 2


