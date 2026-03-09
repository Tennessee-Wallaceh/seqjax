"""Core abstractions and helpers for sequential models.

This module re-exports the primary interfaces for defining models as well as
utilities for simulation and likelihood evaluation so that they are available
under :mod:`seqjax.model`.
"""

from . import ar
from . import interface

ar_model = interface.validate_sequential_model(ar)

__all__ = [
    "ar_model"
]
