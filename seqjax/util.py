import jax
from functools import partial


def index_pytree(tree, index):
    if isinstance(index, int):
        index = (index,)

    def take_index(tree):
        for sub_index in index:
            tree = partial(jax.lax.index_in_dim, index=sub_index, keepdims=False)(tree)
        return tree

    return jax.tree_util.tree_map(take_index, tree)


def slice_pytree(tree, *slice):
    start_index, limit_index = slice

    return jax.tree_util.tree_map(
        partial(jax.lax.slice_in_dim, start_index=start_index, limit_index=limit_index),
        tree,
    )


def pytree_shape(tree):
    # assumes tree is matched and all leaves are arrays
    return jax.tree_leaves(jax.tree_map(lambda t: t.shape, tree))[0]
