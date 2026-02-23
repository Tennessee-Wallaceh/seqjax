"""Utilities for generating Slurm job scripts from Python-defined ablation plans."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from typing import Any


def _quote(token: str) -> str:
    return "'" + token.replace("'", "'\\''") + "'"


def _cartesian_code_grid(axes: dict[str, list[list[str]]]) -> list[tuple[str, ...]]:
    if not axes:
        return [tuple()]

    axis_options = [axes[name] for name in axes]
    combinations: list[tuple[str, ...]] = []
    for option_set in itertools.product(*axis_options):
        merged: list[str] = []
        for bundle in option_set:
            merged.extend(bundle)
        combinations.append(tuple(merged))
    return combinations


def _render_script(
    *,
    experiment_name: str,
    study_name: str,
    output_pattern: str,
    wall_time: str,
    gpus: int,
    source_venv: str,
    install_cmd: str,
    model: str,
    sequence_length: int,
    data_seed: int,
    inference: str,
    fixed_codes: list[str],
    grid_combinations: list[tuple[str, ...]],
    fit_seed_mode: str,
    fixed_fit_seed: int,
    test_samples: int | None,
) -> str:
    max_idx = len(grid_combinations) - 1
    lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH --job-name={study_name}",
        f"#SBATCH --output={output_pattern}",
        f"#SBATCH --gpus={gpus}",
        f"#SBATCH --time={wall_time}",
        f"#SBATCH --array=0-{max_idx}",
        "",
        "set -euo pipefail",
        "",
        f"source {source_venv}",
        install_cmd,
        "",
        "CODES=()",
        'case "$SLURM_ARRAY_TASK_ID" in',
    ]

    for ix, combo in enumerate(grid_combinations):
        lines.append(f"  {ix})")
        for token in combo:
            lines.append(f"    CODES+=({_quote(token)})")
        lines.append("    ;;")

    lines.extend(
        [
            "  *)",
            '    echo "Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2',
            "    exit 1",
            "    ;;",
        ]
    )
    lines.append("esac")
    lines.append("")

    if fit_seed_mode == "task-id":
        lines.append('FIT_SEED="$SLURM_ARRAY_TASK_ID"')
    else:
        lines.append(f"FIT_SEED={fixed_fit_seed}")

    lines.extend(
        [
            "",
            "CMD=(python -m seqjax.cli run",
            f"  --model {model}",
            f"  --sequence-length {sequence_length}",
            f"  --data-seed {data_seed}",
            "  --fit-seed \"$FIT_SEED\"",
            f"  --inference {inference}",
        ]
    )

    if test_samples is not None:
        lines.append(f"  --test-samples {test_samples}")

    for token in fixed_codes:
        lines.append(f"  --code {_quote(token)}")

    lines.append(")")
    lines.append("for code in \"${CODES[@]}\"; do CMD+=(--code \"$code\"); done")
    lines.append(f"CMD+=({_quote(experiment_name)})")
    lines.append('printf "Running: %q " "${CMD[@]}"; echo')
    lines.append('"${CMD[@]}"')

    return "\n".join(lines) + "\n"


def _load_plan(plan_file: str) -> dict[str, Any]:
    path = Path(plan_file)
    spec = importlib.util.spec_from_file_location("seqjax_slurm_plan", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import plan file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PLAN"):
        raise ValueError(f"Plan file {path} must define a top-level PLAN variable")

    plan = getattr(module, "PLAN")
    if not isinstance(plan, dict):
        raise ValueError("PLAN must be a dictionary")

    return plan


def generate_slurm_jobs(
    *,
    plan_file: str,
    experiment_name_override: str | None = None,
    output_root_override: str | None = None,
    dry_run: bool = False,
) -> list[Path]:
    plan = _load_plan(plan_file)

    experiment_name = experiment_name_override or plan["experiment_name"]
    output_root = Path(output_root_override or plan.get("output_root", "experiments/jobs"))
    experiment_dir = output_root / experiment_name
    scripts_dir = experiment_dir / "scripts"
    logs_dir = experiment_dir / "logs"

    shared = dict(plan.get("shared", {}))

    if not dry_run:
        scripts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for study in plan["studies"]:
        study_name = study["name"]
        study_dir_name = str(study_name).replace(" ", "-")

        fixed_codes = list(shared.get("fixed_codes", [])) + list(study.get("fixed_codes", []))
        axes = study.get("axes", {})
        combinations = _cartesian_code_grid(axes)

        output_pattern = str(
            study.get(
                "output_pattern",
                logs_dir / f"{study_dir_name}_%A_%a.out",
            )
        )

        script_text = _render_script(
            experiment_name=experiment_name,
            study_name=str(study.get("job_name", study_dir_name)),
            output_pattern=output_pattern,
            wall_time=str(study.get("wall_time", shared.get("wall_time", "02:30:00"))),
            gpus=int(study.get("gpus", shared.get("gpus", 1))),
            source_venv=str(shared.get("source_venv", ".venv/bin/activate")),
            install_cmd=str(shared.get("install_cmd", "uv pip install -e .[dev]")),
            model=str(shared["model"]),
            sequence_length=int(shared["sequence_length"]),
            data_seed=int(shared["data_seed"]),
            inference=str(shared["inference"]),
            fixed_codes=fixed_codes,
            grid_combinations=combinations,
            fit_seed_mode=str(shared.get("fit_seed_mode", "task-id")),
            fixed_fit_seed=int(shared.get("fixed_fit_seed", 0)),
            test_samples=shared.get("test_samples"),
        )

        path = scripts_dir / f"{study_dir_name}.slurm"
        generated.append(path)
        if not dry_run:
            path.write_text(script_text, encoding="utf-8")

    return generated
