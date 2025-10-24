from __future__ import annotations

import json

from typer.testing import CliRunner

from seqjax.cli import app


runner = CliRunner()


def test_list_models_outputs_registered_label() -> None:
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "simple_stochastic_vol" in result.stdout


def test_show_config_buffer_vi_is_json() -> None:
    result = runner.invoke(app, ["show-config", "buffer-vi"])
    assert result.exit_code == 0
    config = json.loads(result.stdout)
    assert config["method"] == "buffer-vi"


def test_run_dry_run_emits_configuration() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "demo",
            "--model",
            "simple_stochastic_vol",
            "--data-seed",
            "1",
            "--fit-seed",
            "2",
            "--inference",
            "buffer-vi",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["experiment_name"] == "demo"
    assert (
        payload["config"]["data_config"]["target_model_label"]
        == "simple_stochastic_vol"
    )


def test_run_with_shorthand_codes_overrides_buffer_vi_defaults() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "demo",
            "--model",
            "simple_stochastic_vol",
            "--data-seed",
            "1",
            "--fit-seed",
            "2",
            "--inference",
            "buffer-vi",
            "--code",
            "LR-1e-4,MC-10",
            "--code",
            "B-20",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    inference_config = payload["config"]["inference"]["config"]
    assert inference_config["samples_per_context"] == 10
    assert inference_config["buffer_length"] == 20
    assert inference_config["optimization"]["lr"] == 1e-4


def test_run_show_codes_exits_early() -> None:
    result = runner.invoke(app, ["run", "--show-codes"])
    assert result.exit_code == 0
    assert "Buffer VI shorthand codes" in result.stdout
