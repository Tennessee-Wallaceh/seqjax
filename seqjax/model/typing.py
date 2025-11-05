"""Data types for struct of array representations common in jax"""

from functools import lru_cache
import math
from typing import (
    ClassVar,
    TypeVarTuple,
)
import typing
from collections import OrderedDict

import equinox as eqx
import jax
import jax.numpy as jnp


BatchAxes = TypeVarTuple("BatchAxes")


class Packable[*BatchAxes](eqx.Module):
    """
    Mix-in that flattens only the *feature* axis, leaving any leading batch
    axes intact.  Sub-classes provide `_shape_template`, a dict whose values
    are `jax.ShapeDtypeStruct`s **without** batch dims (scalars use shape=()).
    """

    _shape_template: ClassVar[OrderedDict[str, jax.ShapeDtypeStruct]]
    flat_dim: ClassVar[int]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is Packable:
            return

        # ensure subclass provided a template
        template = getattr(cls, "_shape_template", None)
        if not template:
            return

        # compute the flat dim
        cls.flat_dim = sum(
            int(math.prod(spec.shape or (1,))) for spec in template.values()
        )

    @classmethod
    @lru_cache(maxsize=None)
    def _ravel_pair(
        cls,
    ) -> typing.Tuple[
        typing.Callable[["Packable"], jnp.ndarray],
        typing.Callable[[jnp.ndarray], "Packable"],
    ]:
        if cls is Packable or not cls._shape_template:
            raise TypeError(f"{cls.__name__} must define a non-empty _shape_template")

        items = tuple(cls._shape_template.items())
        flat_sizes = tuple(math.prod(spec.shape or (1,)) for _, spec in items)
        split_idx = tuple(sum(flat_sizes[: i + 1]) for i in range(len(flat_sizes) - 1))

        def ravel(obj: "Packable") -> jnp.ndarray:
            parts = []
            for name, spec in items:
                leaf = getattr(obj, name).astype(spec.dtype, copy=False)
                shape = spec.shape
                if shape == ():
                    leaf = leaf[..., None]
                    shape = (1,)
                batch = leaf.shape[: -len(shape)] if shape else leaf.shape
                parts.append(jnp.reshape(leaf, batch + (-1,)))
            return jnp.concatenate(parts, axis=-1)

        def unravel(vec: jnp.ndarray) -> "Packable":
            batch = vec.shape[:-1]
            chunks = jnp.split(vec, split_idx, axis=-1) if split_idx else (vec,)
            kwargs = {}
            for (name, spec), chunk in zip(items, chunks):
                target = spec.shape
                if target == ():
                    # from (...,1) back to scalar trailing shape
                    leaf = jnp.reshape(chunk, batch + (1,))
                    leaf = jnp.reshape(leaf, batch + ())
                else:
                    leaf = jnp.reshape(chunk, batch + target)
                kwargs[name] = leaf.astype(spec.dtype, copy=False)
            return cls(**kwargs)

        return ravel, unravel

    @classmethod
    def ravel(cls, obj: "Packable") -> jnp.ndarray:
        return cls._ravel_pair()[0](obj)

    @classmethod
    def unravel(cls, vec: jnp.ndarray) -> "Packable":
        return cls._ravel_pair()[1](vec)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        # choose the first entry in the template to know feature rank
        first_field = next(iter(self._shape_template))
        feat_rank = len(self._shape_template[first_field].shape)
        leaf = getattr(self, first_field)
        if feat_rank == 0:  # scalar leaf → keep entire shape
            return leaf.shape
        else:
            return leaf.shape[:-feat_rank]

    @classmethod
    def fields(cls):
        return tuple(cls._shape_template.keys())

    @classmethod
    def flat_fields(cls):
        return tuple(
            f"{leaf_name}_{ix}"
            for leaf_name, spec in cls._shape_template.items()
            for ix in range(math.prod(spec.shape))
        )


class Latent[*BatchAxes](Packable[*BatchAxes]): ...


class Observation[*BatchAxes](Packable[*BatchAxes]): ...


class Condition[*BatchAxes](Packable[*BatchAxes]): ...


class NoCondition(Condition):
    __slots__ = ()


class Parameters[*BatchAxes](Packable[*BatchAxes]):
    reference_emission = ()


class HyperParameters[*BatchAxes](Packable[*BatchAxes]): ...
