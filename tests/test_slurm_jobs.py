from __future__ import annotations

import shlex
from pathlib import Path

from _pytest.capture import CaptureFixture
from typer.testing import CliRunner

from seqjax.cli.cli import app
from seqjax.cli import slurm_jobs


def _write_plan(path: Path) -> None:
    path.write_text(
        """
PLAN = {
    "experiment_name": "demo-experiment",
    "output_root": "jobs",
    "shared": {
        "model": "ar-full",
        "inference": "buffer-vi",
        "sequence_length": 25,
        "wall_time": "00:30:00",
        "fit_seed_repeats": 3,
        "data_seed_repeats": 1,
        "base_fit_seed": 10,
        "base_data_seed": 100,
        "fixed_codes": ["SHARED.CODE"],
        "source_venv": ".venv/bin/activate",
        "install_cmd": "echo install",
    },
    "studies": [
        {
            "name": "study_one",
            "axes": {
                "a": [["A.1"], ["A.2"]],
                "b": [["B.1"]],
            },
        }
    ],
}
""",
        encoding="utf-8",
    )


def _extract_cli_argv(script_text: str) -> list[str]:
    lines = script_text.splitlines()
    cmd_start_ix = next(ix for ix, line in enumerate(lines) if line.startswith("CMD=("))
    cmd_end_ix = next(ix for ix in range(cmd_start_ix + 1, len(lines)) if lines[ix] == ")")
    cmd_lines = lines[cmd_start_ix:cmd_end_ix]
    cmd_lines[0] = cmd_lines[0].removeprefix("CMD=(")
    command_tokens = shlex.split(" ".join(line.strip() for line in cmd_lines))
    assert command_tokens[:3] == ["python", "-m", "seqjax.cli"]

    experiment_line = lines[cmd_end_ix + 1]
    assert experiment_line.startswith("CMD+=(") and experiment_line.endswith(")")
    experiment_tokens = shlex.split(experiment_line[len("CMD+=(") : -1])
    argv = [*command_tokens[3:], *experiment_tokens]
    return [
        "101" if token == "$DATA_SEED" else "7" if token == "$FIT_SEED" else token
        for token in argv
    ]


def _drop_code_args(argv: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--code":
            skip_next = True
            continue
        filtered.append(token)
    return filtered


def test_generate_slurm_jobs_one_script_per_configuration(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    plan_file = tmp_path / "plan.py"
    _write_plan(plan_file)

    outputs = slurm_jobs.generate_slurm_jobs(
        plan_file=str(plan_file),
        output_root_override=str(tmp_path / "out"),
        dry_run=False,
    )

    assert [path.name for path in outputs] == ["study_one_0.sh", "study_one_1.sh"]

    script = outputs[0].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in script
    assert 'FIT_SEED="$((BASE_FIT_SEED + (SLURM_ARRAY_TASK_ID % FIT_SEED_REPEATS)))"' in script
    assert 'DATA_SEED="$((BASE_DATA_SEED + (SLURM_ARRAY_TASK_ID / FIT_SEED_REPEATS)))"' in script
    assert '  --data-seed "$DATA_SEED"' in script
    assert '  --fit-seed "$FIT_SEED"' in script
    assert "  --parameters 'base'" in script
    assert "CODES=(" not in script
    assert "for code in \"${CODES[@]}\"" not in script
    assert "  --code 'A.1'" in script
    assert "  --code 'B.1'" in script

    output = capsys.readouterr().out
    assert "Total requested wall time across generated jobs: 03:00:00" in output
    assert "gpu-hours: 3.00" in output

    runner = CliRunner()
    parsed = runner.invoke(app, [*(_drop_code_args(_extract_cli_argv(script))), "--dry-run"])
    assert parsed.exit_code == 0, parsed.stdout


def test_generate_slurm_jobs_latent_fit_mode_script_parses_cli(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.py"
    plan_file.write_text(
        """
PLAN = {
    "experiment_name": "latent-demo",
    "shared": {
        "mode": "latent-fit",
        "model": "ar-full",
        "parameters": "base",
        "sequence_length": 25,
        "num_sequences": 1,
        "fit_seed_repeats": 1,
        "data_seed_repeats": 1,
    },
    "studies": [
        {
            "name": "latent_study",
            "axes": {"a": [["OPT.ADAM", "OPT.ADAM.LR-1e-2"], ["OPT.ADAM", "OPT.ADAM.LR-5e-3"]]},
        }
    ],
}
""",
        encoding="utf-8",
    )

    outputs = slurm_jobs.generate_slurm_jobs(
        plan_file=str(plan_file),
        output_root_override=str(tmp_path / "out"),
        dry_run=False,
    )

    script = outputs[0].read_text(encoding="utf-8")
    assert "CMD=(python -m seqjax.cli latent-fit" in script
    assert "  --inference " not in script
    assert "  --test-samples " not in script

    runner = CliRunner()
    parsed = runner.invoke(app, [*(_extract_cli_argv(script)), "--dry-run"])
    assert parsed.exit_code == 0, parsed.stdout


def test_generate_slurm_jobs_latent_fit_rejects_inference(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.py"
    plan_file.write_text(
        """
PLAN = {
    "experiment_name": "latent-demo",
    "shared": {
        "mode": "latent-fit",
        "model": "aicher_stochastic_vol",
        "parameters": "base",
        "inference": "buffer-vi",
        "sequence_length": 25,
    },
    "studies": [{"name": "latent_study", "axes": {"a": [["OPT.ADAM"]]}}],
}
""",
        encoding="utf-8",
    )

    try:
        slurm_jobs.generate_slurm_jobs(
            plan_file=str(plan_file),
            output_root_override=str(tmp_path / "out"),
            dry_run=True,
        )
    except ValueError as exc:
        assert "does not accept shared.inference" in str(exc)
    else:
        raise AssertionError("Expected ValueError for latent-fit plan with shared.inference")


def test_generate_slurm_jobs_study_overrides_repeats(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.py"
    plan_file.write_text(
        """
PLAN = {
    "experiment_name": "demo",
    "shared": {
        "model": "aicher_stochastic_vol",
        "inference": "buffer-vi",
        "sequence_length": 25,
        "fit_seed_repeats": 3,
        "data_seed_repeats": 1,
    },
    "studies": [
        {
            "name": "study_one",
            "fit_seed_repeats": 2,
            "data_seed_repeats": 2,
            "axes": {"a": [["A.1"]]},
        }
    ],
}
""",
        encoding="utf-8",
    )

    outputs = slurm_jobs.generate_slurm_jobs(
        plan_file=str(plan_file),
        output_root_override=str(tmp_path / "out"),
        dry_run=False,
    )

    script = outputs[0].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3" in script


def test_generate_slurm_jobs_invalid_flat_code_raises(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.py"
    plan_file.write_text(
        """
PLAN = {
    "experiment_name": "demo",
    "shared": {
        "model": "aicher_stochastic_vol",
        "inference": "buffer-vi",
        "sequence_length": 25,
        "fixed_codes": ["BROKEN"],
    },
    "studies": [
        {
            "name": "study_one",
            "axes": {"a": [["A.1"]]},
        }
    ],
}
""",
        encoding="utf-8",
    )

    try:
        slurm_jobs.generate_slurm_jobs(
            plan_file=str(plan_file),
            output_root_override=str(tmp_path / "out"),
            dry_run=True,
        )
    except ValueError as exc:
        assert "Invalid flat code token" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid flat code token")


def test_generate_slurm_jobs_shared_parameters_override(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.py"
    plan_file.write_text(
        """
PLAN = {
    "experiment_name": "demo",
    "shared": {
        "model": "aicher_stochastic_vol",
        "parameters": "challenging",
        "inference": "buffer-vi",
        "sequence_length": 25,
    },
    "studies": [
        {
            "name": "study_one",
            "axes": {"a": [["A.1"]]},
        }
    ],
}
""",
        encoding="utf-8",
    )

    outputs = slurm_jobs.generate_slurm_jobs(
        plan_file=str(plan_file),
        output_root_override=str(tmp_path / "out"),
        dry_run=False,
    )

    script = outputs[0].read_text(encoding="utf-8")
    assert "  --parameters 'challenging'" in script


def test_generate_slurm_jobs_empty_parameters_raises(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.py"
    plan_file.write_text(
        """
PLAN = {
    "experiment_name": "demo",
    "shared": {
        "model": "aicher_stochastic_vol",
        "parameters": "",
        "inference": "buffer-vi",
        "sequence_length": 25,
    },
    "studies": [
        {
            "name": "study_one",
            "axes": {"a": [["A.1"]]},
        }
    ],
}
""",
        encoding="utf-8",
    )

    try:
        slurm_jobs.generate_slurm_jobs(
            plan_file=str(plan_file),
            output_root_override=str(tmp_path / "out"),
            dry_run=True,
        )
    except ValueError as exc:
        assert "shared.parameters must be a non-empty string" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty parameters")


def test_generate_slurm_jobs_mismatched_nested_group_raises(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.py"
    plan_file.write_text(
        """
PLAN = {
    "experiment_name": "demo",
    "shared": {
        "model": "aicher_stochastic_vol",
        "inference": "buffer-vi",
        "sequence_length": 25,
    },
    "studies": [
        {
            "name": "study_one",
            "fixed_codes": ["OPT.ADAM", "OPT.COS.WARM-20"],
            "axes": {"a": [["A.1"]]},
        }
    ],
}
""",
        encoding="utf-8",
    )

    try:
        slurm_jobs.generate_slurm_jobs(
            plan_file=str(plan_file),
            output_root_override=str(tmp_path / "out"),
            dry_run=True,
        )
    except ValueError as exc:
        assert "Mismatched nested code group" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched nested code groups")
