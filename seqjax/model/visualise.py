"""Visualisation utilities for :class:`~seqjax.model.base.SequentialModel`."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Iterable

from graphviz import Digraph

from .base import SequentialModel


def _legend_table(name: str, cls: type) -> str:
    """Return an HTML table for ``cls`` fields."""

    if not is_dataclass(cls):
        return ""
    header = f"<tr><td colspan='2'><b>{name}</b></td></tr>"
    rows = "".join(
        f"<tr><td>{f.name}</td><td>${f.name}$</td></tr>" for f in fields(cls)
    )
    return (
        "<<table border='0' cellborder='1' cellspacing='0'>"
        + header
        + rows
        + "</table>>"
    )


def _add_edges(g: Digraph, srcs: Iterable[str], dst: str) -> None:
    for s in srcs:
        g.edge(s, dst)


def graph_model(model: SequentialModel, *, legend: bool = False) -> Digraph:
    """Return a :class:`graphviz.Digraph` visualising ``model``."""

    g = Digraph("model")
    g.attr(rankdir="LR")

    # parameter node
    g.node("theta", label="θ", shape="square")

    max_order = max(model.prior.order, model.transition.order, model.emission.order)
    start = -max_order + 1

    # latent and observation nodes around t=0 and t=1
    for t in range(start, 2):
        g.node(f"x{t}", label=f"x_{t}")
    for t in range(0, 2):
        g.node(f"y{t}", label=f"y_{t}", shape="doublecircle")

    # prior edges for initial latent states
    for t in range(start, 1):
        g.edge("theta", f"x{t}")

    # transition to x1
    trans_sources = [f"x{1 - i}" for i in range(1, model.transition.order + 1)]
    _add_edges(g, trans_sources, "x1")
    g.edge("theta", "x1")

    # emissions at t=0 and t=1
    for t in range(0, 2):
        lat_srcs = [f"x{t - i}" for i in range(model.emission.order)]
        _add_edges(g, lat_srcs, f"y{t}")
        obs_srcs = [f"y{t - i}" for i in range(1, model.emission.observation_dependency + 1) if t - i >= 0]
        _add_edges(g, obs_srcs, f"y{t}")
        g.edge("theta", f"y{t}")

    if legend:
        orig_args = model.__class__.__orig_bases__[0].__args__
        tables = [
            _legend_table("Particle", orig_args[0]),
            _legend_table("Observation", orig_args[1]),
            _legend_table("Parameters", orig_args[3]),
        ]
        label = "|".join(t for t in tables if t)
        if label:
            g.node("legend", label=label, shape="plaintext")

    return g

