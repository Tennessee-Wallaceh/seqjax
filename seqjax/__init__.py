"""Utilities for sequential probabilistic models built on JAX."""

# Re-export frequently used modules and classes from the package root for
# convenience.  Users can simply ``import seqjax`` and access these components
# without needing to know the underlying module structure.

# simulation and evaluation helpers
from .model import evaluate, simulate

# base model interfaces
from .model.base import Emission, Prior, Target, Transition

# simple inference utilities
from .inference.particlefilter import BootstrapParticleFilter

__all__ = [
    "simulate",
    "evaluate",
    "Prior",
    "Transition",
    "Emission",
    "Target",
    "BootstrapParticleFilter",
]

