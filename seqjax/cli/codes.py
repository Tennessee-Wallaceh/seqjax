"""Utilities for parsing shorthand configuration codes used by the CLI."""

from typing import Any, Callable, TypedDict

from quantiphy import Quantity
from pytimeparse2 import parse as parse_timespan_s # type: ignore[import-untyped]

from seqjax.inference import vi
from seqjax.inference.optimization import registry as optimization_registry
from seqjax.inference.particlefilter import registry as particle_filter_registry
from seqjax.inference.mcmc.metropolis import registry as mcmc_registry
from seqjax.inference.registry import InferenceName

Parser = Callable[[str], Any]

def parse_float(x: str) -> float:
    return float(Quantity(x))

def parse_int_required(x: str) -> int:
    v = float(Quantity(x))
    if int(v) != v:
        raise ValueError(f"Expected integer, got {x!r} -> {v}")
    return int(v)

def parse_int_optional(x: str) -> int | None:
    if x.strip().upper() == "NO":
        return None
    return parse_int_required(x)

def parse_time_optional(x: str) -> int | None:
    if x.strip().upper() == "NO":
        return None
    secs = parse_timespan_s(x)
    if secs is None:
        raise ValueError(f"Bad time string: {x!r}")
    if int(secs) != secs:
        raise ValueError(f"Expected whole seconds, got {x!r} -> {secs}")
    return int(secs)

PARSER_TYPE_LABEL: dict[Parser, str] = {
    parse_float: "float",
    parse_int_required: "integer",
    parse_int_optional: "integer | NO",
    parse_time_optional: "duration | NO",
}

PARSER_EXAMPLES: dict[Parser, list[str]] = {
    parse_float: ["1e-3", "5e-2"],
    parse_int_required: ["500", "1k", "1_000_000"],
    parse_int_optional: ["1k", "NO", "1_000_000"],
    parse_time_optional: ["30m", "2h", "1h30m", "NO"],
}

def format_code_options(
    method_codes: dict[str, Any],
    *,
    examples_by_code: dict[str, list[str]] | None = None,
    max_examples: int = 3,
    path_sep: str = ".",
    kv_sep: str = "-",
    indent: str = "  ",
) -> str:
    """
    Supports leaf specs:
      - (field, parser)
      - (field, parser, example)

    Supports nested specs:
      - {"field": "...", "registry": <ignored>, "options": { "ADAM": ("adam-plain", {...subcodes...}), ...}}
    """

    def _type_label(parser: Any) -> str:
        return PARSER_TYPE_LABEL.get(parser, "value")

    def _raw_examples(codepath: str, parser: Any, example: str | None) -> list[str]:
        # examples_by_code can key on either top-level ("NS") or fully-qualified ("OPT.ADAM.LR")
        if examples_by_code and codepath in examples_by_code:
            return examples_by_code[codepath]
        if example is not None:
            return [example]
        return PARSER_EXAMPLES.get(parser, [])

    def _format_leaf(codepath: str, field: str, parser: Any, example: str | None, level: int) -> list[str]:
        tlabel = _type_label(parser)
        raw = _raw_examples(codepath, parser, example)[:max_examples]
        ex = " | ".join(f"{codepath}{kv_sep}{e}" for e in raw) if raw else ""
        pad = indent * level
        if ex:
            return [f"{pad}{codepath}: {field} ({tlabel}) e.g. {ex}"]
        return [f"{pad}{codepath}: {field} ({tlabel})"]

    def _format_node(node_codes: dict[str, Any], level: int, prefix: str = "") -> list[str]:
        lines: list[str] = []
        for code in sorted(node_codes):
            spec = node_codes[code]
            codepath = f"{prefix}{path_sep}{code}" if prefix else code

            # ---- leaf: (field, parser) or (field, parser, example) ----
            if isinstance(spec, tuple) and len(spec) in (2, 3) and callable(spec[1]):
                field = spec[0]
                parser = spec[1]
                example = spec[2] if len(spec) == 3 else None
                lines.extend(_format_leaf(codepath, field, parser, example, level))
                continue

            # ---- nested: {"field": ..., "options": {...}} ----
            if isinstance(spec, dict) and "field" in spec and "options" in spec:
                field = spec["field"]
                options = spec["options"]
                pad = indent * level

                option_names = sorted(options.keys())
                # Show selector examples (both "OPT-ADAM" and "OPT.ADAM" forms)
                sel_examples = " | ".join(
                    [
                        f"{code}{kv_sep}{opt}"  # OPT-ADAM
                        for opt in option_names[:max_examples]
                    ]
                )
                sel_examples2 = " | ".join(
                    [
                        f"{codepath}{path_sep}{opt}"  # OPT.ADAM
                        for opt in option_names[:max_examples]
                    ]
                )

                lines.append(
                    f"{pad}{codepath}: {field} (choice) options: {', '.join(option_names)}"
                    + (f" e.g. {sel_examples} (or {sel_examples2})" if option_names else "")
                )

                # Expand each option’s subcodes
                for opt in option_names:
                    opt_registry_key, subcodes = options[opt]
                    opt_pad = indent * (level + 1)
                    lines.append(f"{opt_pad}{opt}: {opt_registry_key}")

                    if subcodes:
                        # subcodes are leaf specs keyed by subcode, so recurse with prefix "OPT.ADAM"
                        lines.extend(_format_node(subcodes, level + 2, prefix=f"{codepath}{path_sep}{opt}"))
                continue

            # If you accidentally put something else in the spec, fail loudly.
            raise TypeError(f"Unrecognized spec for code {codepath!r}: {spec!r}")

        return lines

    return "\n".join(_format_node(method_codes, level=0))



FlatCode = tuple[str, Callable[[str], Any], str]
class NestedCode(TypedDict):
    field: str
    registry: Any
    options: dict[str, tuple[str, dict[str, FlatCode]]]

codes: dict[InferenceName, dict[str, FlatCode | NestedCode]] = {}

codes["NUTS"] = {
    "SS": ("step_size", parse_float, "1e-3"),
    "NA": ("num_adaptation", parse_int_required, "5k"),
    "NW": ("num_warmup", parse_int_required, "5k"),
    "NS": ("num_steps", parse_int_optional, "10k"), 
    "NC": ("num_chains", parse_int_required, "1"),
    "MAXT": ("max_time_s", parse_time_optional, "NO"),
}

"""
VI configs
"""
shared_optimizer: dict[str, FlatCode] = {
    "LR": ("lr", parse_float, "1e-3"),
    "MAXS": ("total_steps", parse_int_optional, "NO"),
    "MAXT": ("time_limit_s", parse_time_optional, "5m"),
}

optimization_config: NestedCode = {
    "field": "optimization",
    "registry": optimization_registry.registry,
    "options": {
        "ADAM": ("adam-plain", shared_optimizer)
    }
}

pre_train_config: NestedCode = {
    "field": "pre_training_optimization",
    "registry": optimization_registry.registry,
    "options": {
        "NO": ("none", {}),
        "ADAM": ("adam-plain", shared_optimizer)
    }
}

embedder_config: NestedCode = {
    "field": "embedder",
    "registry": vi.registry.embedder_registry,
    "options": {
        "BiRNN": ("bi-rnn", {
            "H": ("hidden_dim", parse_int_required, "10")
        }),
        "LC": ("long-window", {})
    }
}

codes["full-vi"] = {
    "OPT": optimization_config,
    "MC": ("samples_per_context", parse_int_required, "10"),
    "BS": ("observations_per_step", parse_int_required, "10"),
    "EMB": embedder_config,
    "PAX": {
        "field": "parameter_approximation",
        "registry": vi.registry.parameter_approximation_registry,
        "options": {
            "MF": ("mean-field", {})
        }
    },
    "LAX": {
        "field": "latent_approximation",
        "registry": vi.registry.latent_approximation_registry,
        "options": {
            "SEQ": ("autoregressive", {}),
            "MAF": ("masked-autoregressive-flow", {})
        }
    },
}

codes["buffer-vi"] = codes["full-vi"].copy()
codes["buffer-vi"]["PT"] = pre_train_config
codes["buffer-vi"]["B"] = ("buffer_length", parse_int_required, "10")
codes["buffer-vi"]["M"] = ("batch_length", parse_int_required, "5")

codes["buffer-sgld"] = {
    "SS": ("step_size", parse_float, "1e-3"),
    "NS": ("num_steps", parse_int_optional, "10k"), 
    "MAXT": ("time_limit_s", parse_time_optional, "NO"),
    "B": ("buffer_length", parse_int_required, "10"),
    "M": ("batch_length", parse_int_required, "5"),
    "PF": {
        "field": "particle_filter_config",
        "registry": particle_filter_registry.registry,
        "options": {
            "BTS": ("bootstrap", {
                "N": ("num_particles", parse_int_required, "1k"),
                "R": ("resample", lambda x: x, "multinomial"),
            })
        }
    }
}

codes["full-sgld"] = {
    "SS": ("step_size", parse_float, "1e-3"),
    "NS": ("num_steps", parse_int_optional, "10k"), 
    "MAXT": ("time_limit_s", parse_time_optional, "NO"),
    "PF": {
        "field": "particle_filter_config",
        "registry": particle_filter_registry.registry,
        "options": {
            "BTS": ("bootstrap", {
                "N": ("num_particles", parse_int_required, "1k"),
                "R": ("resample", lambda x: x, "multinomial"),
            })
        }
    }
}

codes["particle-mcmc"] = {
    "MCMC": {
        "field": "mcmc_config",
        "registry": mcmc_registry,
        "options": {
            "MH": ("mh", {
                "SS": ("step_size", parse_float, "1e-3")
            })
        }
    },
    "PF": {
        "field": "particle_filter_config",
        "registry": particle_filter_registry.registry,
        "options": {
            "BTS": ("bootstrap", {
                "N": ("num_particles", parse_int_required, "1k"),
                "R": ("resample", lambda x: x, "multinomial"),
            })
        }
    },
    "MAXT": ("time_limit_s", parse_time_optional, "NO"),
    "NS": ("num_steps", parse_int_optional, "5k"),
}