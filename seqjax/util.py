import jax
from functools import partial
import jax.numpy as jnp

def index_pytree(tree, index):
    if isinstance(index, int):
        index = (index,)

    def take_index(tree):
        for sub_index in index:
            tree = partial(jax.lax.index_in_dim, index=sub_index, keepdims=False)(tree)
        return tree

    return jax.tree_util.tree_map(take_index, tree)

def index_pytree_in_dim(tree, index, dim):

    def take_index(tree):
        return jax.lax.index_in_dim(
            tree, index, axis=dim, keepdims=False
        )
    
    return jax.tree_util.tree_map(take_index, tree)

def dynamic_index_pytree_in_dim(tree, index, dim):

    def take_index(tree):
        return jax.lax.dynamic_index_in_dim(
            tree, index, axis=dim, keepdims=False
        )
    
    return jax.tree_util.tree_map(take_index, tree)


def slice_pytree(tree, start_index, limit_index, dim=0):
    return jax.tree_util.tree_map(
        partial(jax.lax.slice_in_dim, start_index=start_index, limit_index=limit_index, axis=dim),
        tree,
    )

def dynamic_slice_pytree(tree, start_index, limit_index, dim=0):
    return jax.tree_util.tree_map(
        partial(jax.lax.dynamic_slice_in_dim, start_index=start_index, limit_index=limit_index, axis=dim),
        tree,
    )

def promote_scalar_to_vector(x):
    if isinstance(x, jnp.ndarray) and x.ndim == 0:
        return jnp.expand_dims(x, axis=0)
    return x

def concat_pytree(*trees, axis=0):
    promoted_trees = [
        jax.tree_util.tree_map(promote_scalar_to_vector, tree)
        for tree in trees
    ]
    
    return jax.tree_util.tree_map(
        lambda *leaves: jax.lax.concatenate(leaves, dimension=axis),
        *promoted_trees
    )

def pytree_shape(tree):
    # assumes tree is matched and all leaves are arrays
    leaf_shapes = jax.tree.leaves(jax.tree_util.tree_map(lambda t: t.shape, tree))
    return leaf_shapes[0], len(leaf_shapes)

def broadcast_pytree(tree, target_shape):
    def _broadcast(x):
        x = jnp.asarray(x)
        if x.shape == target_shape:
            return x
        elif x.shape == ():  # scalar
            return jnp.broadcast_to(x, target_shape)
        else:
            raise ValueError(f"Expected shape {target_shape} or (), got {x.shape}")
    
    return jax.tree_util.tree_map(_broadcast, tree)

def infer_pytree_shape(pytree):
    leaves, _ = jax.tree_util.tree_flatten(pytree)

    shape = ()
    for x in leaves:
        if jnp.shape(x) != ():
            shape = jnp.shape(x)
            break
        
    return shape