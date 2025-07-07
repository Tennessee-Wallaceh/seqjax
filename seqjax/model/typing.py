"""Runtime type checking utilities for model interfaces."""

import inspect
from typing import Generic, Protocol, TypeVar, TypeVarTuple, Unpack
import typing

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


class Particle(eqx.Module):
    def as_array(self):
        return jnp.dstack(
            [jnp.expand_dims(leaf, -1) for leaf in jax.tree_util.tree_leaves(self)],
        )

    @classmethod
    def from_array(cls, x):
        x_dims = (
            jnp.squeeze(x_dim, -1) for x_dim in jnp.split(x, x.shape[-1], axis=-1)
        )
        return cls(*x_dims)


class Observation(eqx.Module):
    def as_array(self):
        return jnp.dstack(
            [jnp.expand_dims(leaf, -1) for leaf in jax.tree_util.tree_leaves(self)],
        )


class Condition(eqx.Module):
    def as_array(self):
        return jnp.dstack(
            [jnp.expand_dims(leaf, -1) for leaf in jax.tree_util.tree_leaves(self)],
        )


class Parameters(eqx.Module):
    reference_emission = ()

    def as_array(self):
        return jnp.dstack(
            [jnp.expand_dims(leaf, -1) for leaf in jax.tree_util.tree_leaves(self)],
        )


class HyperParameters(eqx.Module): ...


ParticleType = TypeVar("ParticleType", bound=Particle)
ObservationType = TypeVar("ObservationType", bound=Observation)
ConditionType = TypeVar("ConditionType", bound=Condition)
ParametersType = TypeVar("ParametersType", bound=Parameters, contravariant=True)
HyperParametersType = TypeVar("HyperParametersType", bound=HyperParameters)

# Generic helpers -----------------------------------------------------------

Batch = TypeVarTuple("Batch")
SequenceAxis = TypeVar("SequenceAxis", covariant=True)
SampleAxis = TypeVar("SampleAxis", covariant=True)
T_co = TypeVar("T_co", covariant=True)


class Batched(Protocol, Generic[T_co, Unpack[Batch], SequenceAxis]):
    """A :class:`~jaxtyping.PyTree` with arbitrary leading batch axes.

    ``Batch`` represents the shared leading dimensions while ``SequenceAxis``
    denotes the trailing sequence length. ``SampleAxis`` can be used to
    represent batches of independent samples that are not part of the
    sequence dimension. Multiple return values can reuse the same ``Batch``
    tuple to indicate they are batched together.
    """



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
