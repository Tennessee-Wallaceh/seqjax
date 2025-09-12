"""Runtime type checking utilities for model interfaces."""

import inspect
from functools import lru_cache
import math
from typing import (
    ClassVar,
    TypeVar,
    TypeVarTuple,
)
import typing
from collections import OrderedDict

import equinox as eqx
import jax
import jax.numpy as jnp

"""
Suprisingly, Python ABCs ensure that an abstract method is implemented, 
but they don't enforce matching type signature.
To get around this we can use runtime checks to ensure that concrete classes have the 
correct signatures.
We can do this by making the base classes metaclasses, using __init__subclass__ to run checks and ensure:
- all methods are staticmethods (~pure)
- that the implementation signatures match the abstract definitions in generic terms
- that the implementation signatures match order rules
"""

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


class Particle[*BatchAxes](Packable[*BatchAxes]): ...


class Observation[*BatchAxes](Packable[*BatchAxes]): ...


class Condition[*BatchAxes](Packable[*BatchAxes]): ...


class Parameters[*BatchAxes](Packable[*BatchAxes]):
    reference_emission = ()


class HyperParameters[*BatchAxes](Packable[*BatchAxes]): ...


"""
These are the Type Variable versions of the Packable classes
For use in tying input + output types e.g. 
def transition(p: ParticleType) -> ParticleType:
produces the same ParticleType, 
def transition(p: Particle) -> Particle:
could produce any Particle
"""
ParticleType = TypeVar("ParticleType", bound=Particle)
ObservationType = TypeVar("ObservationType", bound=Observation)
ConditionType = TypeVar("ConditionType", bound=Condition)
ParametersType = TypeVar("ParametersType", bound=Parameters)
InferenceParametersType = TypeVar("InferenceParametersType", bound=Parameters)
HyperParametersType = TypeVar("HyperParametersType", bound=HyperParameters)


def resolve_annotation(annotation, type_mapping, class_vars):
    # if the annotation is a type variable, replace it.
    if isinstance(annotation, typing.TypeVar):
        return type_mapping.get(annotation, annotation)

    # replace generic length tuples with specific lengths based on
    # order information
    if (
        typing.get_origin(annotation) is tuple
        and len(annotation.__args__) == 2
        and annotation.__args__[1] is Ellipsis
    ):
        argument_typevar = annotation.__args__[0]
        if argument_typevar == ObservationType:
            order = class_vars["observation_dependency"]
        elif argument_typevar == ParticleType or argument_typevar == ConditionType:
            order = class_vars["order"]
        else:
            raise TypeError(
                f"Unknown order for typevar {argument_typevar} please raise issue.",
            )

        ptype = type_mapping[argument_typevar]
        return annotation.__origin__[tuple(ptype for _ in range(order))]

    return annotation


def normalize_signature(sig, type_mapping, class_vars):
    new_params = []
    for name, param in sig.parameters.items():
        new_annotation = resolve_annotation(param.annotation, type_mapping, class_vars)
        new_params.append(param.replace(annotation=new_annotation))
    new_return = resolve_annotation(sig.return_annotation, type_mapping, class_vars)
    return inspect.Signature(parameters=new_params, return_annotation=new_return)


def check_interface(cls):
    # reference class is the immediate parent
    base = inspect.getmro(cls)[1]
    # print(f"Checking {cls} against {base}")

    # handle generic mapping
    type_mapping = {}
    for gbase in cls.__orig_bases__:
        if hasattr(gbase, "__origin__"):
            type_vars = (
                gbase.__origin__.__parameters__
            )  # e.g. (ParticleType, ParametersType)
            concrete_types = gbase.__args__  # e.g. (LatentVol, LogVolRW)
            type_mapping = {
                **type_mapping,
                **dict(zip(type_vars, concrete_types, strict=False)),
            }
    class_vars = {cvar: getattr(cls, cvar) for cvar in base.__abstractclassvars__}

    # perform the interface checks
    for method_name in ["log_prob", "sample"]:
        base_method = base.__dict__[method_name]
        subclass_method = cls.__dict__.get(method_name, None)

        if subclass_method is None:
            raise TypeError(
                f"{cls.__name__} must override the static method {method_name}.",
            )

        if not isinstance(subclass_method, staticmethod):
            raise TypeError(
                f"In {cls.__name__}, {method_name} must be implemented as a static method.",
            )

        # compare signatures
        base_fn = base_method.__func__
        derived_fn = subclass_method.__func__
        base_sig = inspect.signature(base_fn)
        normalized_base_sig = normalize_signature(base_sig, type_mapping, class_vars)

        derived_sig = inspect.signature(derived_fn)
        if normalized_base_sig != derived_sig:
            raise TypeError(
                f"In {cls.__name__}.{method_name}: signature mismatch\n"
                f"Expected: {normalized_base_sig}\nGot:      {derived_sig} \n"
                f"Class vars: {class_vars} Generic type map: {type_mapping}",
            )


class EnforceInterface:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # skip abstract classes
        if inspect.isabstract(cls):
            return

        # check concrete classes
        check_interface(cls)
