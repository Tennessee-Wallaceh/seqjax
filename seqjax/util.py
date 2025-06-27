import jax
import typing
from functools import partial
import jax.numpy as jnp
from seqjax.model.typing import Batched, Batchable
import haliax as hax
from dataclasses import replace
from jax.experimental import checkify


def haliax_node(x: typing.Any) -> bool:
    return isinstance(x, hax.NamedArray)


def broadcast(batchable: Batchable, axes: tuple[hax.Axis, ...]):
    updates = {}
    for attribute in batchable.attributes:
        updates[attribute] = getattr(batchable, attribute).broadcast_axis(axes)
    return replace(batchable, **updates)


def index_batched(
    batched: Batched[Batchable, tuple[hax.Axis, ...]], index: dict[str, int]
) -> Batchable:
    """Index a Batched object using Haliax-style named indexing.

    Args:
        batched: A Batched struct whose fields are Haliax NamedArrays.
        index: A dict mapping axis names (str) to integer or slice indices.

    Returns:
        A new unbatched struct with indexed fields (e.g. a Particle or Condition).
        Leading axes may be dropped or sliced.
    """

    def take_index(x):
        return x[index]

    return jax.tree_util.tree_map(take_index, batched.value, is_leaf=haliax_node)


def slice_batched(
    batched: Batched[Batchable, tuple[hax.Axis, ...]], slice: dict[str, int]
) -> Batched[Batchable, tuple[hax.Axis, ...]]:
    """Index a Batched object using Haliax-style named indexing.

    Args:
        batched: A Batched struct whose fields are Haliax NamedArrays.
        index: A dict mapping axis names (str) to integer or slice indices.

    Returns:
        A new unbatched struct with indexed fields (e.g. a Particle or Condition).
        Leading axes may be dropped or sliced.
    """

    def take_slice(x):
        return x[slice]

    return Batched(
        jax.tree_util.tree_map(take_slice, batched.value, is_leaf=haliax_node)
    )


def index_pytree_in_dim(tree, index, dim):

    def take_index(tree):
        return jax.lax.index_in_dim(tree, index, axis=dim, keepdims=False)

    return jax.tree_util.tree_map(take_index, tree)


def dynamic_index_pytree_in_dim(tree, index, dim):

    def take_index(tree):
        return jax.lax.dynamic_index_in_dim(tree, index, axis=dim, keepdims=False)

    return jax.tree_util.tree_map(take_index, tree)


def slice_pytree(tree, start_index, limit_index, dim=0):
    return jax.tree_util.tree_map(
        partial(
            jax.lax.slice_in_dim,
            start_index=start_index,
            limit_index=limit_index,
            axis=dim,
        ),
        tree,
    )


def dynamic_slice_pytree(tree, start_index, limit_index, dim=0):
    return jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=start_index,
            limit_index=limit_index,
            axis=dim,
        ),
        tree,
    )


def promote_scalar_to_vector(x):
    if isinstance(x, jnp.ndarray) and x.ndim == 0:
        return jnp.expand_dims(x, axis=0)
    return x


def concat_pytree(*trees, axis=0):
    promoted_trees = [
        jax.tree_util.tree_map(promote_scalar_to_vector, tree) for tree in trees
    ]

    return jax.tree_util.tree_map(
        lambda *leaves: jax.lax.concatenate(leaves, dimension=axis), *promoted_trees
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
