"""Graphviz based visualisation utilities for sequential models."""

from __future__ import annotations

import types
from dataclasses import fields, is_dataclass
from typing import Iterable, Union, get_args, get_origin

from graphviz import Digraph  # type: ignore

from .base import SequentialModel


_UNION_TYPES = tuple(
    t for t in (getattr(types, "UnionType", None), Union) if t is not None
)


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


def _resolve_packable_type(annotation: object) -> type | None:
    """Resolve the concrete class referenced by a type annotation."""

    if annotation is None or annotation is type(None):  # noqa: E721
        return None
    if isinstance(annotation, type):
        return annotation

    origin = get_origin(annotation)

    if origin is tuple:
        args = get_args(annotation)
        return _resolve_packable_type(args[0] if args else None)

    if origin in _UNION_TYPES:
        for arg in get_args(annotation):
            resolved = _resolve_packable_type(arg)
            if resolved is not None:
                return resolved
        return None

    if origin is not None:
        args = get_args(annotation)
        return _resolve_packable_type(args[0] if args else None)

    if hasattr(annotation, "__args__"):
        args = getattr(annotation, "__args__")
        if args:
            return _resolve_packable_type(args[0])

    return None


def graph_model(
    model: SequentialModel,
    *,
    legend: bool = False,
    render: bool | str | None = None,
) -> Digraph:
    """Return a :class:`graphviz.Digraph` visualising ``model``.

    Parameters
    ----------
    model:
        The model to visualise.
    legend:
        If ``True`` then an additional legend describing the particle,
        observation and parameter fields is added.
    render:
        If truthy the graph is rendered using :meth:`graphviz.Digraph.render`.
        If ``True`` the default filename ``model`` is used, otherwise the value
        is interpreted as the filename to render to.
    """

    g = Digraph("model")
    g.attr(rankdir="LR")

    # parameter node
    g.node("theta", label="θ", shape="square")

    particle_cls = getattr(model, "particle_cls", None)
    observation_cls = getattr(model, "observation_cls", None)
    parameter_cls = getattr(model, "parameter_cls", None)
    condition_cls = None

    orig_bases = getattr(model.__class__, "__orig_bases__", ())
    if orig_bases:
        seq_args = get_args(orig_bases[0])
        if len(seq_args) >= 1:
            particle_cls = _resolve_packable_type(seq_args[0]) or particle_cls
        if len(seq_args) >= 5:
            observation_cls = (
                _resolve_packable_type(seq_args[4]) or observation_cls
            )
        if len(seq_args) >= 8:
            condition_cls = _resolve_packable_type(seq_args[7]) or condition_cls
        if len(seq_args) >= 9:
            parameter_cls = _resolve_packable_type(seq_args[8]) or parameter_cls

    if condition_cls is None:
        transition_bases = getattr(model.transition.__class__, "__orig_bases__", ())
        if transition_bases:
            cond_args = get_args(transition_bases[0])
            if len(cond_args) >= 3:
                condition_cls = _resolve_packable_type(cond_args[2]) or condition_cls

    particle_cls = particle_cls or model.particle_cls
    observation_cls = observation_cls or model.observation_cls
    parameter_cls = parameter_cls or model.parameter_cls

    start = -model.prior.order + 1

    particle_fields = (
        [f.name for f in fields(particle_cls)]
        if is_dataclass(particle_cls)
        else ["x"]
    )
    obs_fields = (
        [f.name for f in fields(observation_cls)]
        if is_dataclass(observation_cls)
        else ["y"]
    )
    cond_fields = (
        [f.name for f in fields(condition_cls)]
        if condition_cls is not None and is_dataclass(condition_cls)
        else []
    )

    # create nodes grouped by timestep
    for t in range(start, 2):
        with g.subgraph() as sg:
            sg.attr(rank="same")
            for fld in particle_fields:
                sg.node(f"x{t}_{fld}", label=f"{fld}_{t}")
            if t >= 0:
                for fld in obs_fields:
                    sg.node(f"y{t}_{fld}", label=f"{fld}_{t}", shape="doublecircle")
            if cond_fields:
                for fld in cond_fields:
                    sg.node(f"c{t}_{fld}", label=f"{fld}_{t}")

    # invisible chains for row alignment
    for fields_list, prefix in (
        (particle_fields, "x"),
        (obs_fields, "y"),
        (cond_fields, "c"),
    ):
        for fld in fields_list:
            prev = None
            t_range = range(start, 2)
            if prefix == "y":
                t_range = range(max(start, 0), 2)
            for t in t_range:
                node = f"{prefix}{t}_{fld}"
                if prev is not None:
                    g.edge(prev, node, style="invis")
                prev = node

    # prior edges for initial latent states
    for t in range(start, 1):
        for fld in particle_fields:
            g.edge("theta", f"x{t}_{fld}")
        for cf in cond_fields:
            for pf in particle_fields:
                g.edge(f"c{t}_{cf}", f"x{t}_{pf}")

    # transition to x1
    for fld_dest in particle_fields:
        trans_sources = [
            f"x{1 - i}_{fld_src}"
            for i in range(1, model.transition.order + 1)
            for fld_src in particle_fields
        ]
        _add_edges(g, trans_sources, f"x1_{fld_dest}")
        g.edge("theta", f"x1_{fld_dest}")
        for fld in cond_fields:
            g.edge(f"c1_{fld}", f"x1_{fld_dest}")

    # emissions at t=0 and t=1
    for t in range(0, 2):
        for fld_dest in obs_fields:
            lat_srcs = [
                f"x{t - i}_{fld_src}"
                for i in range(model.emission.order)
                for fld_src in particle_fields
            ]
            _add_edges(g, lat_srcs, f"y{t}_{fld_dest}")

            obs_srcs = [
                f"y{t - i}_{fld_src}"
                for i in range(1, model.emission.observation_dependency + 1)
                if t - i >= 0
                for fld_src in obs_fields
            ]
            _add_edges(g, obs_srcs, f"y{t}_{fld_dest}")
            g.edge("theta", f"y{t}_{fld_dest}")
            for fld in cond_fields:
                g.edge(f"c{t}_{fld}", f"y{t}_{fld_dest}")

    if legend:
        tables = [
            _legend_table("Particle", particle_cls),
            _legend_table("Observation", observation_cls),
            _legend_table("Parameters", parameter_cls),
        ]
        label = "|".join(t for t in tables if t)
        if label:
            g.node("legend", label=label, shape="plaintext")

    if render:
        filename = "model" if render is True else str(render)
        g.render(filename, cleanup=True, format="png")

    return g

