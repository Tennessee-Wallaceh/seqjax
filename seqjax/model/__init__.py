"""Public API for the :mod:`seqjax.model` package.

This module re-exports the most commonly used classes so that projects can
import them directly from :mod:`seqjax.model`.

Example
-------
>>> from seqjax.model import LinearGaussianSSM, LGSSMParameters
>>> model = LinearGaussianSSM()
>>> params = LGSSMParameters()
"""

from seqjax.model.base import SequentialModel
from seqjax.model.visualise import graph_model
from seqjax.model.typing import Observation, Particle
from seqjax.model.linear_gaussian import (
    LinearGaussianSSM,
    LGSSMParameters,
    VectorState,
    VectorObservation,
)

__all__ = [
    "Observation",
    "Particle",
    "SequentialModel",
    "graph_model",
    "LinearGaussianSSM",
    "LGSSMParameters",
    "VectorState",
    "VectorObservation",
]
