"""Command line interface for running ``seqjax`` experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Callable, List, cast

import typer

from seqjax.cli import codes as shorthand_codes
from seqjax.experiment import ExperimentConfig, run_experiment
from seqjax.inference import registry as inference_registry, vi
from seqjax.inference.mcmc import NUTSConfig
from seqjax.model import registry as model_registry

app = typer.Typer(help="Utilities for inspecting and running seqjax experiments.")


def _show_codes_callback(ctx: typer.Context, value: bool) -> bool:
    if value:
        typer.echo("Buffer VI shorthand codes:")
        typer.echo(shorthand_codes.format_buffer_vi_codes())
        typer.echo()
        typer.echo("Full VI shorthand codes:")
        typer.echo(shorthand_codes.format_full_vi_codes())
        raise typer.Exit(code=0)
    return value


def _structure_to_dict(value: Any) -> Any:
    """Recursively convert dataclasses and eqx modules into plain Python types."""

    if is_dataclass(value):
        if isinstance(value, type):
            return value.__name__
        return {k: _structure_to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _structure_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_structure_to_dict(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _structure_to_dict(v) for k, v in vars(value).items()}
    return value


def _normalise_choice(value: str, choices: list[str], what: str) -> str:
    matches = [choice for choice in choices if choice.lower() == value.lower()]
    if not matches:
        raise typer.BadParameter(
            f"Unknown {what} '{value}'. Available choices: {', '.join(sorted(choices))}."
        )
    return matches[0]


def _resolve_model_label(label: str) -> model_registry.SequentialModelLabel:
    canonical = _normalise_choice(
        label,
        sorted(model_registry.sequential_models.keys()),
        "model",
    )
    return cast(model_registry.SequentialModelLabel, canonical)


def _resolve_parameter_label(
    model_label: model_registry.SequentialModelLabel, parameter_label: str
) -> str:
    choices = sorted(model_registry.parameter_settings[model_label].keys())
    return _normalise_choice(parameter_label, choices, "parameter preset")


def _resolve_inference_label(label: str) -> str:
    return _normalise_choice(
        label,
        sorted(inference_registry.inference_functions.keys()),
        "inference method",
    )


InferenceBuilder = Callable[[], inference_registry.InferenceConfig]


def _default_buffer_vi() -> inference_registry.InferenceConfig:
    return inference_registry.BufferVI(
        method="buffer-vi",
        config=vi.BufferedVIConfig(),
    )


def _default_full_vi() -> inference_registry.InferenceConfig:
    return inference_registry.FullVI(
        method="full-vi",
        config=vi.FullVIConfig(),
    )


def _default_nuts() -> inference_registry.InferenceConfig:
    return inference_registry.NUTSInference(
        method="NUTS",
        config=NUTSConfig(),
    )


DEFAULT_INFERENCE_BUILDERS: dict[str, InferenceBuilder] = {
    "buffer-vi": _default_buffer_vi,
    "full-vi": _default_full_vi,
    "NUTS": _default_nuts,
}


@app.command("list-models")
def list_models() -> None:
    """Display registered models and their parameter presets."""

    for label, model_cls in sorted(model_registry.sequential_models.items()):
        presets = ", ".join(sorted(model_registry.parameter_settings[label].keys()))
        typer.echo(f"{label}: {model_cls.__name__} (presets: {presets})")


@app.command("list-inference")
def list_inference() -> None:
    """Display registered inference methods."""

    typer.echo("Available inference methods:")
    for name in sorted(inference_registry.inference_functions.keys()):
        typer.echo(f"  - {name}")
    typer.echo(
        "Defaults available for: "
        + ", ".join(sorted(DEFAULT_INFERENCE_BUILDERS.keys()))
    )


@app.command("show-config")
def show_config(method: str) -> None:
    """Show the default configuration for an inference method."""

    canonical = _resolve_inference_label(method)
    builder = DEFAULT_INFERENCE_BUILDERS.get(canonical)
    if builder is None:
        available = ", ".join(sorted(DEFAULT_INFERENCE_BUILDERS))
        suffix = f" Defaults exist for: {available}." if available else ""
        raise typer.BadParameter(
            f"No default configuration for '{canonical}'.{suffix}"
        )

    config = builder()
    typer.echo(json.dumps(_structure_to_dict(config), indent=2))


@app.command()
def run(
    experiment_name: str = typer.Argument(..., help="W&B project name for the run."),
    model: str = typer.Option(..., "--model", help="Target model label."),
    generative_parameters: str = typer.Option(
        "base", "--parameters", "--params", help="Parameter preset to use."
    ),
    sequence_length: int = typer.Option(
        1024, "--sequence-length", min=1, help="Number of observations to simulate."
    ),
    data_seed: int = typer.Option(..., "--data-seed", help="Seed for data simulation."),
    fit_seed: int = typer.Option(..., "--fit-seed", help="Seed for inference."),
    inference_method: str = typer.Option(..., "--inference", help="Inference method."),
    test_samples: int = typer.Option(
        1000, "--test-samples", min=1, help="Number of posterior samples to draw."
    ),
    code_tokens: List[str] = typer.Option(
        [],
        "--code",
        "-c",
        help=(
            "Shorthand configuration codes like LR-1e-3,MC-10. "
            "Repeat the option or use comma-separated values."
        ),
    ),
    show_codes: bool = typer.Option(
        False,
        "--show-codes",
        callback=_show_codes_callback,
        is_flag=True,
        is_eager=True,
        help="Display the available shorthand codes and exit.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved configuration and exit without running inference.",
    ),
) -> None:
    """Run an experiment using the configured inference method."""

    canonical_model = _resolve_model_label(model)
    canonical_params = _resolve_parameter_label(canonical_model, generative_parameters)
    canonical_inference = _resolve_inference_label(inference_method)

    data_config = model_registry.DataConfig(
        target_model_label=canonical_model,
        generative_parameter_label=canonical_params,
        sequence_length=sequence_length,
        seed=data_seed,
    )

    builder = DEFAULT_INFERENCE_BUILDERS.get(canonical_inference)
    if builder is None:
        available = ", ".join(sorted(DEFAULT_INFERENCE_BUILDERS))
        suffix = f" Defaults exist for: {available}." if available else ""
        raise typer.BadParameter(
            "No default configuration is available for "
            f"'{canonical_inference}'.{suffix}"
        )

    inference_config_obj = builder()

    if code_tokens:
        if inference_config_obj.method == "buffer-vi":
            try:
                configured_buffer = shorthand_codes.apply_buffer_vi_codes(
                    inference_config_obj.config,
                    code_tokens,
                )
            except shorthand_codes.CodeParseError as exc:
                raise typer.BadParameter(str(exc)) from exc
            inference_config_obj = replace(
                inference_config_obj, config=configured_buffer
            )
        elif inference_config_obj.method == "full-vi":
            try:
                configured_full = shorthand_codes.apply_full_vi_codes(
                    inference_config_obj.config,
                    code_tokens,
                )
            except shorthand_codes.CodeParseError as exc:
                raise typer.BadParameter(str(exc)) from exc
            inference_config_obj = replace(
                inference_config_obj, config=configured_full
            )
        else:
            raise typer.BadParameter(
                "Shorthand codes are currently only supported for the buffer-vi "
                "and full-vi methods."
            )

    experiment_config = ExperimentConfig(
        data_config=data_config,
        test_samples=test_samples,
        fit_seed=fit_seed,
        inference=inference_config_obj,
    )

    if dry_run:
        payload = {
            "experiment_name": experiment_name,
            "config": _structure_to_dict(experiment_config),
        }
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=0)

    typer.echo(
        f"Running experiment '{experiment_name}' with model '{canonical_model}' "
        f"and inference '{canonical_inference}'."
    )
    run_experiment(experiment_name, experiment_config)


def main() -> None:
    """Entry point used by the console script."""

    app()


__all__ = ["app", "main", "run", "list_models", "list_inference", "show_config"]
