from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generate_slurm_jobs_command_avoids_experiment_import(tmp_path: Path) -> None:
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
    "studies": [{"name": "study", "axes": {"a": [["A.1"]]}}],
}
""",
        encoding="utf-8",
    )

    script = f"""
import sys
from seqjax.cli import cli
print('before_experiment=' + str('seqjax.experiment' in sys.modules))
cli.generate_slurm_jobs_cmd(plan_file=r'{plan_file}', experiment_name=None, output_root=None, dry_run=True)
print('after_experiment=' + str('seqjax.experiment' in sys.modules))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "before_experiment=False" in completed.stdout
    assert "after_experiment=False" in completed.stdout
