"""Utility helpers for manipulating JAX pytrees."""

from functools import partial
from typing import Any, Iterable

import jax
import jax.numpy as jnp


def index_pytree(tree: Any, index: int | Iterable[int]) -> Any:
    """Index a pytree of arrays along the first dimension."""

    if isinstance(index, int):
        index = (index,)

    def take_index(tree: Any) -> Any:
        for sub_index in index:
            tree = partial(jax.lax.index_in_dim, index=sub_index, keepdims=False)(tree)
        return tree

    return jax.tree_util.tree_map(take_index, tree)


def index_pytree_in_dim(tree: Any, index: int, dim: int) -> Any:
    """Index a pytree along a specified dimension."""

    def take_index(tree: Any) -> Any:
        return jax.lax.index_in_dim(
            tree,
            index,
            axis=dim,
            keepdims=False,
        )

    return jax.tree_util.tree_map(take_index, tree)


def dynamic_index_pytree_in_dim(tree: Any, index: int, dim: int) -> Any:
    """Dynamically index a pytree along a specified dimension."""

    def take_index(tree: Any) -> Any:
        return jax.lax.dynamic_index_in_dim(
            tree,
            index,
            axis=dim,
            keepdims=False,
        )

    return jax.tree_util.tree_map(take_index, tree)


def slice_pytree(tree: Any, start_index: int, limit_index: int, dim: int = 0) -> Any:
    """Slice a pytree along ``dim`` between ``start_index`` and ``limit_index``."""

    return jax.tree_util.tree_map(
        partial(
            jax.lax.slice_in_dim,
            start_index=start_index,
            limit_index=limit_index,
            axis=dim,
        ),
        tree,
    )


def dynamic_slice_pytree(
    tree: Any,
    start_index: int,
    limit_index: int,
    dim: int = 0,
) -> Any:
    """Dynamically slice a pytree along ``dim``."""

    slice_size = limit_index - start_index
    return jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=start_index,
            slice_size=slice_size,
            axis=dim,
        ),
        tree,
    )


def concat_pytree(*trees: Any, axis: int = 0) -> Any:
    """Concatenate pytrees along ``axis``."""
    def _concat(*leaves):
        max_ndim = max(leaf.ndim for leaf in leaves)
        expanded = [
            jnp.expand_dims(leaf, list(range(max_ndim - leaf.ndim)))
            if leaf.ndim < max_ndim
            else leaf
            for leaf in leaves
        ]
        return jax.lax.concatenate(expanded, dimension=axis)

    return jax.tree_util.tree_map(_concat, *trees)


def pytree_shape(tree: Any) -> tuple[tuple[int, ...], int]:
    """Return the shape of leaves and leaf count of ``tree``."""

    # assumes tree is matched and all leaves are arrays
    leaves = jax.tree_util.tree_leaves(tree)
    leaf_shapes = [jnp.shape(leaf) for leaf in leaves]
    return leaf_shapes[0], len(leaf_shapes)


def broadcast_pytree(tree: Any, target_shape: tuple[int, ...]) -> Any:
    """Broadcast leaves in ``tree`` to ``target_shape``."""

    def _broadcast(x: Any) -> Any:
        x = jnp.asarray(x)
        if x.shape == target_shape:
            return x
        if x.shape == ():  # scalar
            return jnp.broadcast_to(x, target_shape)
        raise ValueError(f"Expected shape {target_shape} or (), got {x.shape}")

    return jax.tree_util.tree_map(_broadcast, tree)


def infer_pytree_shape(pytree: Any) -> tuple[int, ...]:
    """Infer a broadcastable shape from a pytree."""

    leaves, _ = jax.tree_util.tree_flatten(pytree)

    shape: tuple[int, ...] = ()
    for x in leaves:
        if jnp.shape(x) != ():
            shape = jnp.shape(x)
            break

    return shape
