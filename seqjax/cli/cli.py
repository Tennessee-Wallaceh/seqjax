"""Command line interface for running ``seqjax`` experiments."""
import json
import typing
from dataclasses import asdict, is_dataclass
import typer

from seqjax.cli import codes
from seqjax.cli import slurm_jobs
from seqjax.experiment import ExperimentConfig, RuntimeConfig, run_experiment
from seqjax.inference import registry as inference_registry
from seqjax.model import registry as model_registry
from .results import ResultProcessor


app = typer.Typer(help="Utilities for inspecting and running seqjax experiments.")

StorageMode = typing.Literal["wandb", "wandb-offline"]

def _resolve_model_label(label: str) -> model_registry.SequentialModelLabel:
    if label not in model_registry.posterior_factories:
        typer.echo(f"Model {label} not found.")
        raise Exception(f"Model {label} not found.")

    return typing.cast(model_registry.SequentialModelLabel, label)


def _resolve_inference_label(label: str) -> inference_registry.InferenceName:
    if label not in inference_registry.inference_registry:
        typer.echo(f"Inference {label} not found.")
        raise Exception(f"Inference {label} not found.")

    return typing.cast(inference_registry.InferenceName, label)

def _structure_to_dict(value: typing.Any) -> typing.Any:
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

def parse_straight_codes(code_tokens, available_codes):
    dict_config = {}
    parsed_codes = set()
    for code_token in code_tokens:
        code, option = code_token.split("-", 1)
        field, parser, _ = available_codes[code]
        dict_config[field] = parser(option)
        parsed_codes.add(code)

    for code, options in available_codes.items():
        if code not in parsed_codes and not isinstance(options, dict):
            (field, parser, default) = options
            dict_config[field] = parser(default)

    return dict_config

def build_inference_config(
    method: inference_registry.InferenceName, 
    code_tokens: list[str]
) ->  inference_registry.InferenceConfig:
    try:
        available_codes = codes.codes[method]
    except KeyError:
        raise typer.BadParameter(
            f"Inference method {method} has no configured codes."
        )
    
    # read off the straight forward config
    straight_code_tokens = [token for token in code_tokens if "." not in token]
    dict_config = parse_straight_codes(straight_code_tokens, available_codes)

    # firstly need to unflatten the nested codes
    nested_code_tokens = [token for token in code_tokens if "." in token]
    nested_code_groups: dict[str, list[str]] = {}
    nested_code_value = {}
    for code_token in nested_code_tokens:
        nested_code_config = code_token.split(".")
        if len(nested_code_config) == 2:
            code, group_value = nested_code_config
            sub_code = None
        else:
            code, group_value, sub_code = nested_code_config
        
        if code not in nested_code_value:
            nested_code_value[code] = group_value
            nested_code_groups[code] = []
        else:
            assert nested_code_value[code] == group_value, f"Mismatched options for {code}!"

        if sub_code is not None:
            nested_code_groups[code].append(sub_code)

    # then can process them 
    parsed_nested_codes = set()
    for code, sub_codes in nested_code_groups.items():
        subconfig = typing.cast(codes.NestedCode, available_codes[code])
        selected_option, selected_option_codes = subconfig["options"][nested_code_value[code]]
        subconfig_dict = parse_straight_codes(sub_codes, selected_option_codes)
        dict_config[subconfig["field"]] = subconfig["registry"][selected_option](
            **subconfig_dict
        )
        parsed_nested_codes.add(code)

    # then build defaults for any nested code not provided
    available_nested_codes = [
        token for token in available_codes 
        if isinstance(available_codes[token], dict) and token not in parsed_nested_codes
    ]
    for code in available_nested_codes:
        subconfig = typing.cast(codes.NestedCode, available_codes[code])
        selected_code = next(iter(subconfig["options"]))
        selected_option, selected_option_codes = subconfig["options"][selected_code]
        dict_config[subconfig["field"]] = subconfig["registry"][selected_option](
            **parse_straight_codes([], selected_option_codes)
        )

    return inference_registry.inference_registry[method].build_config(dict_config)

@app.command("list-models")
def list_models() -> None:
    """Display registered models and their parameter presets."""

    for label, model in sorted(model_registry.posterior_factories.items()):
        presets = ", ".join(sorted(model_registry.parameter_settings[label].keys()))
        typer.echo(f"{label}: {model.__class__.__name__} (presets: {presets})")

@app.command("list-inference")
def list_inference() -> None:
    """Display registered inference methods."""

    typer.echo("Available inference methods:")
    for name in sorted(inference_registry.inference_registry):
        typer.echo(f"  - {name}")

@app.command("list-codes")
def list_codes(method: inference_registry.InferenceName) -> None:
    try:
        typer.echo(codes.format_code_options(codes.codes[method]))
    except KeyError:
        typer.echo(f"Inference method {method} not found.")



@app.command("generate-slurm-jobs")
def generate_slurm_jobs_cmd(
    plan_file: str = typer.Option(..., "--plan-file", "--plan", help="Python plan file defining PLAN dict."),
    experiment_name: str | None = typer.Option(None, "--experiment-name", help="Override plan experiment name."),
    output_root: str | None = typer.Option(None, "--output-root", help="Override plan output root folder."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve and print target files without writing."),
) -> None:
    """Generate Slurm scripts from a Python-defined ablation plan."""

    outputs = slurm_jobs.generate_slurm_jobs(
        plan_file=plan_file,
        experiment_name_override=experiment_name,
        output_root_override=output_root,
        dry_run=dry_run,
    )

    if dry_run:
        typer.echo("Dry-run: would generate the following scripts:")
    else:
        typer.echo("Generated scripts:")

    for output in outputs:
        typer.echo(f"  - {output}")

@app.command()
def run(
    experiment_name: str = typer.Argument(..., help="W&B project name for the run."),
    model: str = typer.Option(..., "--model", help="Target model label."),
    generative_parameters: str = typer.Option(
        "base", "--parameters", "--params", help="Parameter preset to use."
    ),
    sequence_length: int = typer.Option(
        1000, "--sequence-length", min=1, help="Number of observations to simulate."
    ),
    num_sequences: int = typer.Option(
        1,
        "--num-sequences",
        min=1,
        help="Number of independent observation sequences to simulate.",
    ),
    data_seed: int = typer.Option(..., "--data-seed", help="Seed for data simulation."),
    fit_seed: int = typer.Option(..., "--fit-seed", help="Seed for inference."),
    inference_method: str = typer.Option(..., "--inference", help="Inference method."),
    test_samples: int = typer.Option(
        1000, "--test-samples", min=1, help="Number of posterior samples to draw."
    ),
    code_tokens: typing.List[str] = typer.Option(
        [],
        "--code",
        "-c",
        help=(
            "Shorthand configuration codes like LR-1e-3,MC-10. "
            "Repeat the option or use comma-separated values."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved configuration and exit without running inference.",
    ),
    storage_mode: StorageMode = typer.Option(
        "wandb",
        "--storage-mode",
        help="Storage backend mode: wandb uploads remotely, wandb-offline stores locally.",
    ),
    local_root: str = typer.Option(
        "./wandb",
        "--local-root",
        help="Local directory used by W&B offline mode.",
    ),
) -> None:
    """Run an experiment using the configured inference method."""

    canonical_model = _resolve_model_label(model)
    canonical_method = _resolve_inference_label(inference_method)

    data_config = model_registry.DataConfig(
        target_model_label=canonical_model,
        generative_parameter_label=generative_parameters,
        sequence_length=sequence_length,
        num_sequences=num_sequences,
        seed=data_seed,
    )

    inference_config = build_inference_config(canonical_method, code_tokens)

    experiment_config = ExperimentConfig(
        data_config=data_config,
        test_samples=test_samples,
        fit_seed=fit_seed,
        inference=inference_config,
    )

    if dry_run:
        payload = {
            "experiment_name": experiment_name,
            "config": _structure_to_dict(experiment_config),
            "runtime": {
                "storage_mode": storage_mode,
                "local_root": local_root,
            },
        }
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=0)

    typer.echo(
        f"Running experiment '{experiment_name}' with model '{canonical_model}' "
        f"and inference '{inference_method}'."
    )
    run_experiment(
        experiment_name,
        experiment_config,
        ResultProcessor(),
        runtime_config=RuntimeConfig(storage_mode=storage_mode, local_root=local_root),
    )


def main() -> None:
    """Entry point used by the console script."""

    app()


__all__ = ["app", "main", "run", "list_models", "list_inference", "show_config"]
