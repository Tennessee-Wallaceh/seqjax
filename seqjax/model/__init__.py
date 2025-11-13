"""Core abstractions and helpers for sequential models.

This module re-exports the primary interfaces for defining models as well as
utilities for simulation and likelihood evaluation so that they are available
under :mod:`seqjax.model`.
"""

from . import evaluate as _evaluate_module
from . import simulate as _simulate_module
from .base import (
    BayesianSequentialModel,
    Emission,
    ParameterPrior,
    Prior,
    SequentialModel,
    Transition,
)

# Re-export the submodules so ``from seqjax.model import evaluate`` continues to
# return the documented modules.
evaluate = _evaluate_module
simulate = _simulate_module

# Convenience aliases for the most commonly used helpers.
log_prob_x = _evaluate_module.log_prob_x
log_prob_y_given_x = _evaluate_module.log_prob_y_given_x
log_prob_joint = _evaluate_module.log_prob_joint
simulate_sequence = _simulate_module.simulate
simulate_step = _simulate_module.step

__all__ = [
    "evaluate",
    "simulate",
    "BayesianSequentialModel",
    "Emission",
    "ParameterPrior",
    "Prior",
    "SequentialModel",
    "Transition",
    "log_prob_x",
    "log_prob_y_given_x",
    "log_prob_joint",
    "simulate_sequence",
    "simulate_step",
]
