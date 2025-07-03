"""Utilities for sequential probabilistic models built on JAX."""

# Re-export frequently used modules and classes from the package root for
# convenience.  Users can simply ``import seqjax`` and access these components
# without needing to know the underlying module structure.

# simulation and evaluation helpers
from .model import evaluate, simulate
from .model.visualise import graph_model

# base model interfaces
from .model.base import Emission, Prior, SequentialModel, Transition

# simple inference utilities
from .inference.particlefilter import BootstrapParticleFilter

# Maintain backwards compatibility with the old ``Target`` name.
Target = SequentialModel

__all__ = [
    "simulate",
    "evaluate",
    "Prior",
    "Transition",
    "Emission",
    "SequentialModel",
    "Target",
    "graph_model",
    "BootstrapParticleFilter",
]

