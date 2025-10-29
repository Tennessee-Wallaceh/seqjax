"""Utilities for parsing shorthand configuration codes used by the CLI."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Tuple

from seqjax.inference.vi import run as vi_run

__all__ = [
    "CodeParseError",
    "BUFFER_VI_CODES",
    "BUFFER_VI_DEFAULT_CODES",
    "FULL_VI_CODES",
    "FULL_VI_DEFAULT_CODES",
    "format_buffer_vi_codes",
    "format_full_vi_codes",
    "apply_buffer_vi_codes",
    "apply_full_vi_codes",
]


class CodeParseError(ValueError):
    """Raised when shorthand configuration codes cannot be parsed."""


_CodeValue = Any | Callable[[], Any]


def _evaluate(value: _CodeValue) -> Any:
    if callable(value):
        return value()
    return value


def _merge_code_definitions(
    *code_sets: Dict[str, Dict[str, _CodeValue]],
) -> Dict[str, Dict[str, _CodeValue]]:
    merged: Dict[str, Dict[str, _CodeValue]] = {}
    for code_set in code_sets:
        for key, entries in code_set.items():
            merged[key] = dict(entries)
    return merged


COMMON_CODE_DEFINITIONS: Dict[str, Dict[str, _CodeValue]] = {
    "LR": {"1e-2": 1e-2, "1e-3": 1e-3, "1e-4": 1e-4},
    "MC": {"1": 1, "10": 10, "50": 50, "100": 100},
    "BS": {"1": 1, "10": 10, "50": 50, "100": 100},
    "EMB": {
        "LC": lambda: vi_run.LongContextEmbedder(prev_window=5, post_window=5),
        "SC": lambda: vi_run.ShortContextEmbedder(prev_window=2, post_window=2),
        "BiRNN": lambda: vi_run.BiRNNEmbedder(hidden_dim=5),
    },
}

BUFFER_SPECIFIC_CODES: Dict[str, Dict[str, _CodeValue]] = {
    "PT": {"N": 0, "Y": 5000},
    "B": {"5": 5, "20": 20},
    "M": {"5": 5, "20": 20},
}

FULL_SPECIFIC_CODES: Dict[str, Dict[str, _CodeValue]] = {}


BUFFER_VI_CODES: Dict[str, Dict[str, _CodeValue]] = _merge_code_definitions(
    COMMON_CODE_DEFINITIONS,
    BUFFER_SPECIFIC_CODES,
)


FULL_VI_CODES: Dict[str, Dict[str, _CodeValue]] = _merge_code_definitions(
    COMMON_CODE_DEFINITIONS,
    FULL_SPECIFIC_CODES,
)


COMMON_FACTOR_NAMES: Dict[str, str] = {
    "LR": "learning_rate",
    "MC": "mc_samples",
    "BS": "minibatch_size",
    "EMB": "embedding",
}

BUFFER_SPECIFIC_FACTOR_NAMES: Dict[str, str] = {
    "PT": "pretrain_steps",
    "B": "buffer_len",
    "M": "batch_len",
}

FULL_SPECIFIC_FACTOR_NAMES: Dict[str, str] = {}


BUFFER_VI_FACTOR_NAMES: Dict[str, str] = {
    **COMMON_FACTOR_NAMES,
    **BUFFER_SPECIFIC_FACTOR_NAMES,
}


FULL_VI_FACTOR_NAMES: Dict[str, str] = {
    **COMMON_FACTOR_NAMES,
    **FULL_SPECIFIC_FACTOR_NAMES,
}


COMMON_DEFAULT_CODES: List[str] = [
    "LR-1e-3",
    "MC-1",
    "BS-1",
    "EMB-SC",
]


BUFFER_VI_DEFAULT_CODES: List[str] = [
    *COMMON_DEFAULT_CODES,
    "PT-N",
    "B-5",
    "M-5",
]


FULL_VI_DEFAULT_CODES: List[str] = [
    *COMMON_DEFAULT_CODES,
]

# Backwards compatibility for older imports
DEFAULT_CODES = BUFFER_VI_DEFAULT_CODES


def _parse_token(
    token: str, code_definitions: Dict[str, Dict[str, _CodeValue]]
) -> Tuple[str, Any]:
    if "-" not in token:
        raise CodeParseError(
            f"Bad code '{token}'. Expected PREFIX-SUFFIX like 'LR-1e-3'."
        )
    prefix, suffix = token.split("-", 1)
    if prefix not in code_definitions:
        valid = ", ".join(sorted(code_definitions))
        raise CodeParseError(f"Unknown factor '{prefix}'. Valid factors: {valid}")
    sub = code_definitions[prefix]
    if suffix not in sub:
        raise CodeParseError(
            f"Invalid value '{suffix}' for factor '{prefix}'. Choices: {', '.join(sub.keys())}"
        )
    return prefix, _evaluate(sub[suffix])


def _normalise_tokens(raw_tokens: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for raw in raw_tokens:
        if not raw:
            continue
        tokens.extend([token.strip() for token in raw.split(",") if token.strip()])
    return tokens


def _resolve_config(
    tokens: Iterable[str],
    *,
    defaults: Iterable[str],
    code_definitions: Dict[str, Dict[str, _CodeValue]],
    factor_names: Dict[str, str],
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for default in defaults:
        key, value = _parse_token(default, code_definitions)
        resolved[factor_names.get(key, key)] = value

    for token in tokens:
        key, value = _parse_token(token, code_definitions)
        resolved[factor_names.get(key, key)] = value

    return resolved


def format_buffer_vi_codes() -> str:
    lines: List[str] = []
    for factor, entries in sorted(BUFFER_VI_CODES.items()):
        values = ", ".join(f"{factor}-{suffix}" for suffix in entries)
        lines.append(f"{factor}: {values}")
    return "\n".join(lines)


def format_full_vi_codes() -> str:
    lines: List[str] = []
    for factor, entries in sorted(FULL_VI_CODES.items()):
        values = ", ".join(f"{factor}-{suffix}" for suffix in entries)
        lines.append(f"{factor}: {values}")
    return "\n".join(lines)


def apply_buffer_vi_codes(
    config: vi_run.BufferedVIConfig, raw_tokens: Iterable[str]
) -> vi_run.BufferedVIConfig:
    tokens = _normalise_tokens(raw_tokens)
    try:
        resolved = _resolve_config(
            tokens,
            defaults=BUFFER_VI_DEFAULT_CODES,
            code_definitions=BUFFER_VI_CODES,
            factor_names=BUFFER_VI_FACTOR_NAMES,
        )
    except CodeParseError as exc:  # pragma: no cover - exercised via CLI
        raise exc

    learning_rate = resolved["learning_rate"]
    optimization = config.optimization
    if isinstance(optimization, vi_run.AdamOpt):
        optimization = replace(optimization, lr=learning_rate)
    else:  # pragma: no cover - future-proofing
        raise CodeParseError("Shorthand learning rate codes require Adam optimization.")

    embedder = resolved["embedding"]
    if not isinstance(embedder, vi_run.EmbedderConfig):
        raise CodeParseError("Invalid embedder resolved from shorthand codes.")

    return replace(
        config,
        optimization=optimization,
        buffer_length=resolved["buffer_len"],
        batch_length=resolved["batch_len"],
        observations_per_step=resolved["minibatch_size"],
        samples_per_context=resolved["mc_samples"],
        pre_training_steps=resolved["pretrain_steps"],
        embedder=embedder,
    )


def apply_full_vi_codes(
    config: vi_run.FullVIConfig, raw_tokens: Iterable[str]
) -> vi_run.FullVIConfig:
    tokens = _normalise_tokens(raw_tokens)
    try:
        resolved = _resolve_config(
            tokens,
            defaults=FULL_VI_DEFAULT_CODES,
            code_definitions=FULL_VI_CODES,
            factor_names=FULL_VI_FACTOR_NAMES,
        )
    except CodeParseError as exc:  # pragma: no cover - exercised via CLI
        raise exc

    learning_rate = resolved["learning_rate"]
    optimization = config.optimization
    if isinstance(optimization, vi_run.AdamOpt):
        optimization = replace(optimization, lr=learning_rate)
    else:  # pragma: no cover - future-proofing
        raise CodeParseError("Shorthand learning rate codes require Adam optimization.")

    embedder = resolved["embedding"]
    if not isinstance(embedder, vi_run.EmbedderConfig):
        raise CodeParseError("Invalid embedder resolved from shorthand codes.")

    return replace(
        config,
        optimization=optimization,
        observations_per_step=resolved["minibatch_size"],
        samples_per_context=resolved["mc_samples"],
        embedder=embedder,
    )
