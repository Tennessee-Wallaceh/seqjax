from __future__ import annotations

import dataclasses
from abc import ABCMeta
from collections.abc import Callable, Hashable
from dataclasses import Field as DataclassField
from typing import Any, TypeVar
from typing import dataclass_transform

_ModuleT = TypeVar("_ModuleT", bound="Module")


def field(
    *,
    converter: Callable[[Any], Any] | None = ...,
    static: bool = ...,
    **kwargs: Any,
) -> DataclassField[Any]: ...


@dataclass_transform(field_specifiers=(dataclasses.field, field))
class _ModuleMeta(ABCMeta):
    def __call__(cls: type[_ModuleT], *args: Any, **kwargs: Any) -> _ModuleT: ...


class Module(Hashable, metaclass=_ModuleMeta):
    def __hash__(self) -> int: ...

    def __eq__(self, other: Any) -> bool: ...
