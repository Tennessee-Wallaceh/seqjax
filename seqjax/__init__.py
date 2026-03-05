"""Utilities for sequential probabilistic models built on JAX."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "simulate",
    "evaluate",
    "graph_model",
    "Prior",
    "Transition",
    "Proposal",
    "Emission",
    "SequentialModel",
    "InferenceMethod",
]


def __getattr__(name: str) -> Any:
    if name in {"simulate", "evaluate"}:
        model_module = import_module("seqjax.model")
        return getattr(model_module, name)

    if name == "graph_model":
        visualise_module = import_module("seqjax.model.visualise")
        return visualise_module.graph_model

    if name in {"Emission", "Prior", "SequentialModel", "Transition"}:
        base_module = import_module("seqjax.model.base")
        return getattr(base_module, name)

    if name == "Proposal":
        particle_filter_module = import_module("seqjax.inference.particlefilter")
        return particle_filter_module.Proposal

    if name == "InferenceMethod":
        interface_module = import_module("seqjax.inference.interface")
        return interface_module.InferenceMethod

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
