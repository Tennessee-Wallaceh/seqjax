"""Utilities for generating Slurm job scripts from Python-defined ablation plans."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from typing import Any, TypedDict, cast


class SharedPlan(TypedDict, total=False):
    model: str
    sequence_length: int
    num_sequences: int
    inference: str
    test_samples: int
    wall_time: str
    gpus: int
    source_venv: str
    install_cmd: str
    fixed_codes: list[str]
    fit_seed_repeats: int
    data_seed_repeats: int
    base_fit_seed: int
    base_data_seed: int


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
    num_sequences: int,
    base_data_seed: int,
    inference: str,
    fixed_codes: list[str],
    combination: tuple[str, ...],
    fit_seed_repeats: int,
    data_seed_repeats: int,
    base_fit_seed: int,
    test_samples: int | None,
) -> str:
    total_repeats = fit_seed_repeats * data_seed_repeats
    if total_repeats <= 0:
        raise ValueError("fit_seed_repeats * data_seed_repeats must be positive")

    max_idx = total_repeats - 1
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
        f"FIT_SEED_REPEATS={fit_seed_repeats}",
        f"DATA_SEED_REPEATS={data_seed_repeats}",
        f"BASE_FIT_SEED={base_fit_seed}",
        f"BASE_DATA_SEED={base_data_seed}",
        "",
        'FIT_SEED="$((BASE_FIT_SEED + (SLURM_ARRAY_TASK_ID % FIT_SEED_REPEATS)))"',
        'DATA_SEED="$((BASE_DATA_SEED + (SLURM_ARRAY_TASK_ID / FIT_SEED_REPEATS)))"',
        "",
        "CODES=(",
    ]

    for token in combination:
        lines.append(f"  {_quote(token)}")

    lines.extend(
        [
            ")",
            "",
            "CMD=(python -m seqjax.cli run",
            f"  --model {model}",
            f"  --sequence-length {sequence_length}",
            f"  --num-sequences {num_sequences}",
            '  --data-seed "$DATA_SEED"',
            '  --fit-seed "$FIT_SEED"',
            f"  --inference {inference}",
        ]
    )

    if test_samples is not None:
        lines.append(f"  --test-samples {test_samples}")

    for token in fixed_codes:
        lines.append(f"  --code {_quote(token)}")

    lines.append(")")
    lines.append('for code in "${CODES[@]}"; do CMD+=(--code "$code"); done')
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


def _parse_hhmmss(raw: str) -> int:
    h, m, s = raw.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _format_hhmmss(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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

    shared = cast(SharedPlan, dict(plan.get("shared", {})))

    if not dry_run:
        scripts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    total_requested_seconds = 0
    total_gpu_seconds = 0

    for study in plan["studies"]:
        study_name = study["name"]
        study_dir_name = str(study_name).replace(" ", "-")

        fixed_codes = list(shared.get("fixed_codes", [])) + list(study.get("fixed_codes", []))
        axes = study.get("axes", {})
        combinations = _cartesian_code_grid(axes)

        wall_time = str(study.get("wall_time", shared.get("wall_time", "02:30:00")))
        gpus = int(study.get("gpus", shared.get("gpus", 1)))
        fit_seed_repeats = int(study.get("fit_seed_repeats", shared.get("fit_seed_repeats", 3)))
        data_seed_repeats = int(study.get("data_seed_repeats", shared.get("data_seed_repeats", 1)))
        base_fit_seed = int(study.get("base_fit_seed", shared.get("base_fit_seed", 0)))
        base_data_seed = int(study.get("base_data_seed", shared.get("base_data_seed", 0)))

        output_pattern = str(
            study.get(
                "output_pattern",
                logs_dir / f"{study_dir_name}_%A_%a.out",
            )
        )

        per_run_seconds = _parse_hhmmss(wall_time)
        repeat_count = fit_seed_repeats * data_seed_repeats

        for job_ix, combination in enumerate(combinations):
            script_text = _render_script(
                experiment_name=experiment_name,
                study_name=str(study.get("job_name", study_dir_name)),
                output_pattern=output_pattern,
                wall_time=wall_time,
                gpus=gpus,
                source_venv=str(shared.get("source_venv", ".venv/bin/activate")),
                install_cmd=str(shared.get("install_cmd", "uv pip install -e .[dev]")),
                model=str(shared["model"]),
                sequence_length=int(shared["sequence_length"]),
                num_sequences=int(shared.get("num_sequences", 1)),
                base_data_seed=base_data_seed,
                inference=str(shared["inference"]),
                fixed_codes=fixed_codes,
                combination=combination,
                fit_seed_repeats=fit_seed_repeats,
                data_seed_repeats=data_seed_repeats,
                base_fit_seed=base_fit_seed,
                test_samples=shared.get("test_samples"),
            )

            path = scripts_dir / f"{study_dir_name}_{job_ix}.sh"
            generated.append(path)
            if not dry_run:
                path.write_text(script_text, encoding="utf-8")

            total_requested_seconds += per_run_seconds * repeat_count
            total_gpu_seconds += per_run_seconds * repeat_count * gpus

    print(
        "Total requested wall time across generated jobs: "
        f"{_format_hhmmss(total_requested_seconds)} (gpu-hours: {total_gpu_seconds / 3600:.2f})"
    )

    return generated
