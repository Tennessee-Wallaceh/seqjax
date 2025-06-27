"""
Builds out typing infrastructure for the package.
The key idea is to define a data containers as eqx.Modules, then keep track of the batch dimensions via haliax.

Python ABCs ensure that an abstract method is implemented,
but they don't enforce matching type signature.
To get around this we can use runtime checks to ensure that concrete classes have the
correct signatures.
We can do this by making the base classes metaclasses, using __init__subclass__ to run checks and ensure:
- all methods are staticmethods (~pure)
- that the implementation signatures match the abstract definitions in generic terms
- that the implementation signatures match order rules
"""

import typing
import inspect
import jax.numpy as jnp
import equinox as eqx
import jax
from dataclasses import dataclass, replace, fields
from typing import Generic, Union
import haliax as hax


class _Struct(eqx.Module):
    def __init_subclass__(cls):
        for field_name, _ in cls.__annotations__.items():
            if not field_name.startswith("_") and not hasattr(
                cls, f"_spec_{field_name}"
            ):
                raise TypeError(
                    f"Particle subclass '{cls.__name__}' is missing axis spec: _spec_{field_name}"
                )

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(name for name in self.__annotations__ if not name.startswith("_"))

    def get_spec(self, attribute) -> hax.Axis:
        return getattr(self, f"_spec_{attribute}")


class Particle(_Struct):
    pass


class Observation(_Struct):
    pass


class Condition(_Struct):
    pass


class Parameters(eqx.Module):
    reference_emission = ()

    def as_array(self):
        return jnp.dstack(
            [jnp.expand_dims(l, -1) for l in jax.tree_util.tree_leaves(self)]
        )


class HyperParameters(eqx.Module): ...


ParticleType = typing.TypeVar("ParticleType", bound=Particle)
ObservationType = typing.TypeVar("ObservationType", bound=Observation)
ConditionType = typing.TypeVar("ConditionType", bound=Condition)
ParametersType = typing.TypeVar("ParametersType", bound=Parameters, contravariant=True)
HyperParametersType = typing.TypeVar("HyperParametersType", bound=HyperParameters)
BatchAxes = typing.TypeVar("BatchAxes", bound=tuple[hax.Axis, ...])
Batchable = typing.TypeVar("Batchable", bound=Union[Condition, Particle, Observation])

PathAxis = typing.TypeVar("PathAxis", bound=hax.Axis)


def infer_batch_axes(obj: _Struct) -> BatchAxes:
    axes = None
    for field in fields(obj):
        if field.name.startswith("_"):
            continue
        val = getattr(obj, field.name)
        if not isinstance(val, hax.NamedArray):
            continue
        spec = getattr(obj, f"_spec_{field.name}", ())
        if not isinstance(spec, tuple):
            spec = (spec,)
        batch_axes = val.axes[: len(val.axes) - len(spec)]
        if axes is None:
            axes = batch_axes
        elif axes != batch_axes:
            raise ValueError(f"Inconsistent batch axes: {axes} vs {batch_axes}")
    if axes is None:
        raise ValueError("No NamedArray fields found")
    return axes


@dataclass
class Batched(Generic[Batchable, BatchAxes]):
    """A wrapper that adds shared leading axes to a structured particle.

    `Batched` represents a particle type whose fields are individually structured
    with named axes (defined by `_spec_<field>` attributes), and which also share
    a common set of leading axes such as 'batch' or 'time'.

    The total axes for each field are the concatenation of:
        - `axes`: shared leading axes (e.g., batch, sequence),
        - `_spec_<field>`: field-specific axes defined on the `Batchable` class.

    This allows clear separation between model-internal structure and batching layout,
    enabling named-axis operations (e.g., summing, indexing, mapping) to be applied
    safely and consistently.

    Attributes:
        value: A structured particle instance (e.g., a dataclass of NamedArrays).
        axes: A tuple of Haliax axes shared by all fields in `value`.

    """

    value: Batchable

    @property
    def batch_axes(self) -> BatchAxes:
        return infer_batch_axes(self.value)


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
        elif argument_typevar == ParticleType:
            order = class_vars["order"]
        elif argument_typevar == ConditionType:
            order = class_vars["order"]
        else:
            raise TypeError(
                f"Unknown order for typevar {argument_typevar} please raise issue."
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
            type_mapping = {**type_mapping, **dict(zip(type_vars, concrete_types))}
    class_vars = {cvar: getattr(cls, cvar) for cvar in base.__abstractclassvars__}

    # perform the interface checks
    for method_name in ["log_p", "sample"]:
        base_method = base.__dict__[method_name]
        subclass_method = cls.__dict__.get(method_name, None)

        if subclass_method is None:
            raise TypeError(
                f"{cls.__name__} must override the static method {method_name}."
            )

        if not isinstance(subclass_method, staticmethod):
            raise TypeError(
                f"In {cls.__name__}, {method_name} must be implemented as a static method."
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
                f"Class vars: {class_vars} Generic type map: {type_mapping}"
            )


class EnforceInterface:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # skip abstract classes
        if inspect.isabstract(cls):
            return

        # check concrete classes
        check_interface(cls)
