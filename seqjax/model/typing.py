"""Data types for struct of array representations common in jax"""

from functools import lru_cache
import math
from typing import (
    ClassVar,
    TypeVarTuple,
    get_origin,
)
import typing
from collections import OrderedDict
import dataclasses
import jax
import jax.numpy as jnp
import equinox as eqx

BatchAxes = TypeVarTuple("BatchAxes")

class classproperty:
    def __init__(self, fget):
        self.fget = fget
    def __get__(self, obj, cls):
        return self.fget(cls)
    

class Packable[*BatchAxes](eqx.Module):
    """
    Mix-in that flattens only the *feature* axis, leaving any leading batch
    axes intact.  Sub-classes provide `_shape_template`, a dict whose values
    are `jax.ShapeDtypeStruct`s **without** batch dims (scalars use shape=()).
    """

    _shape_template: ClassVar[OrderedDict[str, jax.ShapeDtypeStruct]]
    flat_dim: ClassVar[int]

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is Packable or abstract:
            return

        # ensure subclass provided a template
        template = getattr(cls, "_shape_template", None)
        if template is None:
            raise TypeError(
                f"{cls.__qualname__} is Packable so must define _shape_template"
            )

        if not abstract:
            dataclasses.dataclass()(cls)

        # check the fields against the template
        field_names = {
            name
            for name, tp in cls.__annotations__.items()
            if not name.startswith("_") and get_origin(tp) is not ClassVar
        }
        template_names = set(template.keys())
        missing = field_names - template_names
        extra = template_names - field_names
        if missing or extra:
            raise TypeError(
                f"{cls.__qualname__}: _shape_template mismatch; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        # compute the flat dim
        cls.flat_dim = sum(
            int(math.prod(spec.shape or (1,))) for spec in template.values()
        )

    @classproperty
    def flat_sizes(cls):
        return tuple(
            math.prod(spec.shape or (1,))
            for spec in cls._shape_template.values()
        )
    
    def ravel(self) -> jnp.ndarray:
        parts = []
        for name, spec in self._shape_template.items():
            leaf = getattr(self, name).astype(spec.dtype, copy=False)
            shape = spec.shape
            if shape == ():
                leaf = leaf[..., None]
                shape = (1,)
            batch = leaf.shape[: -len(shape)] if shape else leaf.shape
            parts.append(jnp.reshape(leaf, batch + (-1,)))
        if len(parts) == 0:
            return jnp.zeros(self.batch_shape + (0,), dtype=jnp.float32)
        else:
            return jnp.concatenate(parts, axis=-1)
    
    @classmethod
    def unravel(cls, vec: jnp.ndarray) -> typing.Self:
        split_idx = tuple(
            sum(cls.flat_sizes[: i + 1]) 
            for i in range(len(cls.flat_sizes) - 1)
        )

        batch = vec.shape[:-1]
        chunks = jnp.split(vec, split_idx, axis=-1) if split_idx else (vec,)
        kwargs = {}
        for (name, spec), chunk in zip(
            cls._shape_template.items(), 
            chunks
        ):
            target = spec.shape
            if target == ():
                # from (...,1) back to scalar trailing shape
                leaf = jnp.reshape(chunk, batch + (1,))
                leaf = jnp.reshape(leaf, batch + ())
            else:
                leaf = jnp.reshape(chunk, batch + target)
            kwargs[name] = leaf.astype(spec.dtype, copy=False)

        return cls(**kwargs)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        # choose the first entry in the template to know feature rank
        if len(self._shape_template) == 0:
            return ()

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


class Latent[*BatchAxes](Packable[*BatchAxes], abstract=True): ...


class Observation[*BatchAxes](Packable[*BatchAxes], abstract=True): ...


class Condition[*BatchAxes](Packable[*BatchAxes], abstract=True): ...


class NoCondition(Condition):
    __slots__ = ()
    _shape_template = OrderedDict()


class Parameters[*BatchAxes](Packable[*BatchAxes], abstract=True): ...


class HyperParameters[*BatchAxes](Packable[*BatchAxes], abstract=True): ...
