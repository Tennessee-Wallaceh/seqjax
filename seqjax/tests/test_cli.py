from __future__ import annotations

import json

from typer.testing import CliRunner

from seqjax.cli import app


TWO_HOURS_SECONDS = 2 * 60 * 60
FIVE_HOURS_SECONDS = 5 * 60 * 60
HALF_HOUR_SECONDS = 30 * 60
FIFTY_THOUSAND_STEPS = 50_000
MILLION_STEPS = 1_000_000


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
    assert config["config"]["optimization"]["time_limit_s"] == TWO_HOURS_SECONDS


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
    assert (
        payload["config"]["inference"]["config"]["optimization"]["time_limit_s"]
        == TWO_HOURS_SECONDS
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
    assert inference_config["optimization"]["time_limit_s"] == TWO_HOURS_SECONDS


def test_run_with_rl_code_sets_long_time_limit_for_buffer_vi() -> None:
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
            "RL-L",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    time_limit = (
        payload["config"]["inference"]["config"]["optimization"]["time_limit_s"]
    )
    assert time_limit == FIVE_HOURS_SECONDS


def test_run_with_ts_code_sets_steps_for_buffer_vi() -> None:
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
            "TS-H",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    total_steps = (
        payload["config"]["inference"]["config"]["optimization"]["total_steps"]
    )
    assert total_steps == FIFTY_THOUSAND_STEPS


def test_run_with_shorthand_codes_overrides_full_vi_defaults() -> None:
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
            "full-vi",
            "--code",
            "LR-1e-4,MC-10",
            "--code",
            "BS-10,EMB-LC",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    inference_config = payload["config"]["inference"]["config"]
    assert inference_config["samples_per_context"] == 10
    assert inference_config["observations_per_step"] == 10
    assert inference_config["optimization"]["lr"] == 1e-4
    assert inference_config["embedder"]["label"] == "long-window"
    assert inference_config["optimization"]["time_limit_s"] == TWO_HOURS_SECONDS


def test_run_with_rl_code_sets_short_time_limit_for_full_vi() -> None:
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
            "full-vi",
            "--code",
            "RL-S",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    time_limit = (
        payload["config"]["inference"]["config"]["optimization"]["time_limit_s"]
    )
    assert time_limit == HALF_HOUR_SECONDS


def test_run_with_ts_code_sets_steps_for_full_vi() -> None:
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
            "full-vi",
            "--code",
            "TS-L",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    total_steps = (
        payload["config"]["inference"]["config"]["optimization"]["total_steps"]
    )
    assert total_steps == MILLION_STEPS


def test_run_show_codes_exits_early() -> None:
    result = runner.invoke(app, ["run", "--show-codes"])
    assert result.exit_code == 0
    assert "Buffer VI shorthand codes" in result.stdout
    assert "Full VI shorthand codes" in result.stdout
