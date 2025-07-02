import os
import sys
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from seqjax.util import pytree_shape


def test_pytree_shape_dict():
    tree = {'a': jnp.zeros((2, 3)), 'b': jnp.ones((2, 3))}
    shape, leaf_count = pytree_shape(tree)
    assert shape == (2, 3)
    assert leaf_count == 2


