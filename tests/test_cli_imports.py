from __future__ import annotations

import subprocess
import sys


def test_importing_cli_module_does_not_import_jax() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import seqjax.cli.cli; print('jax' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "False"
