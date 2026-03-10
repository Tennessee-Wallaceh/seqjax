"""Core abstractions and helpers for sequential models.

This module re-exports the primary interfaces for defining models as well as
utilities for simulation and likelihood evaluation so that they are available
under :mod:`seqjax.model`.
"""

import importlib

from . import ar
from . import interface

_typing = importlib.import_module("typing")

ar_model: interface.SequentialModelProtocol = interface.validate_sequential_model(
    _typing.cast(interface.SequentialModelProtocol, ar)
)

__all__ = ["ar_model"]
