from __future__ import annotations

from pathlib import Path

from _pytest.capture import CaptureFixture

from seqjax.cli import slurm_jobs


def _write_plan(path: Path) -> None:
    path.write_text(
        """
PLAN = {
    "experiment_name": "demo-experiment",
    "output_root": "jobs",
    "shared": {
        "model": "aicher_stochastic_vol",
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
    assert "CODES=(" not in script
    assert "for code in \"${CODES[@]}\"" not in script
    assert "  --code 'A.1'" in script
    assert "  --code 'B.1'" in script

    output = capsys.readouterr().out
    assert "Total requested wall time across generated jobs: 03:00:00" in output
    assert "gpu-hours: 3.00" in output


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


def test_generate_slurm_jobs_supports_sub_grid_references(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.py"
    plan_file.write_text(
        """
PLAN = {
    "experiment_name": "demo",
    "shared": {
        "model": "aicher_stochastic_vol",
        "inference": "buffer-vi",
        "sequence_length": 25,
        "sub_grids": {
            "embedder_variants": [["EMB.C1D.H-2"], ["EMB.C1D.H-4"]],
        },
    },
    "studies": [
        {
            "name": "study_one",
            "sub_grids": {
                "latent_variants": [["LAX.MAF.W-20"], ["LAX.MAF.W-32"]],
            },
            "axes": {
                "model": [
                    {
                        "codes": ["EMB.C1D", "LAX.MAF"],
                        "sub_grids": ["embedder_variants", "latent_variants"],
                    }
                ],
            },
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

    assert [path.name for path in outputs] == [
        "study_one_0.sh",
        "study_one_1.sh",
        "study_one_2.sh",
        "study_one_3.sh",
    ]

    first_script = outputs[0].read_text(encoding="utf-8")
    last_script = outputs[-1].read_text(encoding="utf-8")
    assert "  --code 'EMB.C1D.H-2'" in first_script
    assert "  --code 'LAX.MAF.W-20'" in first_script
    assert "  --code 'EMB.C1D.H-4'" in last_script
    assert "  --code 'LAX.MAF.W-32'" in last_script


def test_generate_slurm_jobs_errors_for_unknown_sub_grid(tmp_path: Path) -> None:
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
            "axes": {
                "model": [
                    {"codes": ["EMB.C1D"], "sub_grid": "missing_grid"},
                ],
            },
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
            dry_run=False,
        )
    except ValueError as exc:
        assert "Unknown sub_grid 'missing_grid'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown sub-grid reference")


def test_generate_slurm_jobs_respects_start_ix(tmp_path: Path) -> None:
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
            "start_ix": 7,
            "axes": {"a": [["A.1"], ["A.2"]]},
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

    assert [path.name for path in outputs] == ["study_one_7.sh", "study_one_8.sh"]


def test_generate_slurm_jobs_rejects_invalid_start_ix(tmp_path: Path) -> None:
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
            "start_ix": -1,
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
            dry_run=False,
        )
    except ValueError as exc:
        assert "start_ix must be >= 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid start_ix")
