"""Command line interface for running ``seqjax`` experiments."""
import json
import typing
from dataclasses import asdict, dataclass, is_dataclass
import typer

from seqjax.cli import slurm_jobs
from seqjax.experiment import make_record_trigger

if typing.TYPE_CHECKING:
    from seqjax.inference import registry as inference_registry
    from seqjax.model import registry as model_registry

app = typer.Typer(help="Utilities for inspecting and running seqjax experiments.")

StorageMode = typing.Literal["wandb", "wandb-offline"]
DataSource = typing.Literal["synthetic", "real"]

def _resolve_model_label(label: str) -> "model_registry.SequentialModelLabel":
    from seqjax.model import registry as model_registry

    if label not in model_registry.posterior_factories:
        typer.echo(f"Model {label} not found.")
        raise Exception(f"Model {label} not found.")

    return typing.cast("model_registry.SequentialModelLabel", label)


def _resolve_inference_label(label: str) -> str:
    from seqjax.inference import registry as inference_registry

    if label not in inference_registry.inference_registry:
        typer.echo(f"Inference {label} not found.")
        raise Exception(f"Inference {label} not found.")

    return label

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


@dataclass(frozen=True)
class LatentInferenceConfig:
    optimization: typing.Any
    embedder: typing.Any
    latent_approximation: typing.Any
    batch_length: int
    samples_per_context: int
    unroll: int
    compiled_steps: int

    @property
    def label(self) -> str:
        return self.latent_approximation.label


def build_latent_inference_config(code_tokens: list[str]) -> LatentInferenceConfig:
    inference_config = build_inference_config("buffer-vi", code_tokens)
    vi_config = inference_config.config
    return LatentInferenceConfig(
        optimization=vi_config.optimization,
        embedder=vi_config.embedder,
        latent_approximation=vi_config.latent_approximation,
        batch_length=vi_config.batch_length,
        samples_per_context=vi_config.samples_per_context,
        unroll=vi_config.unroll,
        compiled_steps=vi_config.compiled_steps,
    )

def build_inference_config(
    method: str,
    code_tokens: list[str]
) -> typing.Any:
    from seqjax.cli import codes
    from seqjax.inference import registry as inference_registry

    canonical_method = typing.cast("inference_registry.InferenceName", method)

    try:
        available_codes = codes.codes[canonical_method]
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
        nested_code_config = code_token.split(".", 2)
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
    return inference_registry.inference_registry[canonical_method].build_config(dict_config)

@app.command("list-models")
def list_models() -> None:
    """Display registered models and their parameter presets."""
    from seqjax.model import registry as model_registry

    for label, model in sorted(model_registry.posterior_factories.items()):
        presets = ", ".join(sorted(model_registry.parameter_settings[label].keys()))
        typer.echo(f"{label}: {model.__class__.__name__} (presets: {presets})")

@app.command("list-inference")
def list_inference() -> None:
    """Display registered inference methods."""
    from seqjax.inference import registry as inference_registry

    typer.echo("Available inference methods:")
    for name in sorted(inference_registry.inference_registry):
        typer.echo(f"  - {name}")

@app.command("list-codes")
def list_codes(method: str) -> None:
    from seqjax.cli import codes

    try:
        canonical_method = typing.cast("inference_registry.InferenceName", method)
        typer.echo(codes.format_code_options(codes.codes[canonical_method]))
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
    model: str | None = typer.Option(None, "--model", help="Target model label."),
    generative_parameters: str | None = typer.Option(
        "base", "--parameters", "--params", help="Parameter preset to use."
    ),
    sequence_length: int | None = typer.Option(
        1000, "--sequence-length", min=1, help="Number of observations to simulate."
    ),
    num_sequences: int | None = typer.Option(
        1,
        "--num-sequences",
        min=1,
        help="Number of independent observation sequences to simulate.",
    ),
    data_seed: int | None = typer.Option(None, "--data-seed", help="Seed for data simulation."),
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
    data_source: DataSource = typer.Option(
        "synthetic",
        "--data-source",
        help="Data loading mode: synthetic (default) or real.",
    ),
    dataset_name: str | None = typer.Option(
        None,
        "--dataset-name",
        help="Prepared dataset name to load when --data-source=real.",
    ),
    data_root: str = typer.Option(
        "./",
        "--data-root",
        help="Root directory for prepared datasets when --data-source=real.",
    ),
    init_from_generative: bool = typer.Option(
        False,
        "--true-init",
        help="Whether to use the generative param+latent to initialize.",
    ),
) -> None:
    """Run an experiment using the configured inference method."""
    from seqjax import io
    from seqjax.experiment import ExperimentConfig, RuntimeConfig, run_experiment
    from seqjax.model import registry as model_registry
    from .results import ResultProcessor

    canonical_method = _resolve_inference_label(inference_method)
    canonical_model: typing.Any
    data_config: typing.Any

    if data_source == "synthetic":
        if dataset_name is not None:
            raise typer.BadParameter(
                "--dataset-name can only be used with --data-source=real."
            )
        if model is None:
            raise typer.BadParameter("--model is required when --data-source=synthetic.")

        if generative_parameters is None:
            raise typer.BadParameter(
                "--parameters is required when --data-source=synthetic."
            )
        
        if sequence_length is None:
            raise typer.BadParameter(
                "--sequence-length is required when --data-source=synthetic."
            )
        if data_seed is None:
            raise typer.BadParameter(
                "--data-seed is required when --data-source=synthetic."
            )
        if num_sequences is None:
            num_sequences = 1

        canonical_model = _resolve_model_label(model)

        data_config = model_registry.SyntheticDataConfig(
            target_model_label=canonical_model,
            generative_parameter_label=generative_parameters,
            sequence_length=sequence_length,
            num_sequences=num_sequences,
            seed=data_seed,
        )


    elif data_source == "real":
        if dataset_name is None:
            raise typer.BadParameter(
                "--dataset-name is required when --data-source=real."
            )
        storage_backend = io.LocalFilesystemDataStorage(data_root)
        manifest = storage_backend.load_manifest(dataset_name) 
        canonical_model = _resolve_model_label(manifest['model_label'])
        data_config = model_registry.RealDataConfig(
            dataset_name=dataset_name,
            target_model_label=canonical_model,
            sequence_length=manifest['sequence_length'],
            num_sequences=manifest['num_sequences'],
        )

    inference_config = build_inference_config(canonical_method, code_tokens)

    experiment_config = ExperimentConfig(
        data_config=data_config,
        test_samples=test_samples,
        fit_seed=fit_seed,
        inference=inference_config,
        init_from_generative=init_from_generative,
    )

    if dry_run:
        payload = {
            "experiment_name": experiment_name,
            "config": _structure_to_dict(experiment_config),
            "runtime": {
                "storage_mode": storage_mode,
                "local_root": local_root,
                "data_source": data_source,
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
        runtime_config=RuntimeConfig(
            storage_mode=storage_mode, 
            local_root=local_root, 
            data_root=data_root if data_source == "real" else None,
        ),
    )


def main() -> None:
    """Entry point used by the console script."""

    app()


@app.command("latent-fit")
def latent_fit(
    experiment_name: str = typer.Argument(..., help="W&B project name for the run."),
    model: str | None = typer.Option(None, "--model", help="Target model label."),
    generative_parameters: str | None = typer.Option(
        "base", "--parameters", "--params", help="Parameter preset to use."
    ),
    sequence_length: int | None = typer.Option(
        1000, "--sequence-length", min=1, help="Number of observations to simulate."
    ),
    num_sequences: int | None = typer.Option(
        1,
        "--num-sequences",
        min=1,
        help="Number of independent observation sequences to simulate.",
    ),
    data_seed: int | None = typer.Option(None, "--data-seed", help="Seed for data simulation."),
    fit_seed: int = typer.Option(..., "--fit-seed", help="Seed for inference."),
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
        help="Print the resolved configuration and exit without fitting.",
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
    """Fit a latent variational approximation on synthetic data only."""
    from dataclasses import asdict

    import jax.random as jrandom
    import wandb

    from seqjax import io
    from seqjax.experiment import RuntimeConfig, configure_wandb_runtime
    from seqjax.inference.interface import ObservationDataset
    from seqjax.inference.optimization import registry as optimization_registry
    from seqjax.inference.vi import registry as vi_registry
    from seqjax.inference.vi import train_latent
    from seqjax.model import registry as model_registry
    import seqjax.model.typing as seqjtyping
    from .latent_fit import WandbLatentFitArtifactSink

    if model is None:
        raise typer.BadParameter("--model is required.")
    if generative_parameters is None:
        raise typer.BadParameter("--parameters is required.")
    if sequence_length is None:
        raise typer.BadParameter("--sequence-length is required.")
    if data_seed is None:
        raise typer.BadParameter("--data-seed is required.")
    if num_sequences is None:
        num_sequences = 1
    if num_sequences != 1:
        raise typer.BadParameter(
            "latent-fit currently supports exactly one sequence (--num-sequences=1)."
        )

    canonical_model = _resolve_model_label(model)

    data_config = model_registry.SyntheticDataConfig(
        target_model_label=canonical_model,
        generative_parameter_label=generative_parameters,
        sequence_length=sequence_length,
        num_sequences=num_sequences,
        seed=data_seed,
    )

    latent_inference_config = build_latent_inference_config(code_tokens)

    if isinstance(latent_inference_config.optimization, optimization_registry.NoOpt):
        raise typer.BadParameter(
            "latent-fit requires an optimizer; received OPT.NO from --code."
        )

    runtime_config = RuntimeConfig(storage_mode=storage_mode, local_root=local_root)

    if dry_run:
        payload = {
            "experiment_name": experiment_name,
            "mode": "latent-fit",
            "data_config": _structure_to_dict(data_config),
            "latent_inference_config": _structure_to_dict(latent_inference_config),
            "runtime": _structure_to_dict(runtime_config),
        }
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=0)

    storage = io.LocalFilesystemDataStorage(local_root)
    _x_paths, observation_paths, conditions = storage.get_data(data_config)

    condition_paths = seqjtyping.NoCondition() if conditions is None else conditions
    dataset = ObservationDataset(
        observations=typing.cast(seqjtyping.Observation, observation_paths),
        conditions=typing.cast(seqjtyping.Condition, condition_paths),
    )

    target = data_config.target
    params = data_config.generative_parameters

    key = jrandom.key(fit_seed)
    key, embed_key, latent_key, train_key = jrandom.split(key, 4)

    embed = vi_registry._build_embedder(
        latent_inference_config.embedder,
        target,
        seqjtyping.NoParam,
        dataset.sequence_length,
        dataset.sequence_length,
        embed_key,
    )
    latent_approximation = vi_registry.build_latent_approximation(
        latent_inference_config.latent_approximation,
        sample_length=dataset.sequence_length,
        target_model=target,
        key=latent_key,
        latent_context_dims=embed.latent_context_dims,
    )

    optim = optimization_registry.build_optimizer(latent_inference_config.optimization)
    latent_sampling_kwargs: typing.Any = {
        "mc-samples": latent_inference_config.samples_per_context
    }


    """
    build tracker connecting to wandb
    """
    configure_wandb_runtime(runtime_config)
    wandb_run = typing.cast(
        io.WandbRun,
        wandb.init(
            project=experiment_name,
            config={
                "mode": "latent-fit",
                "data_config": asdict(data_config),
                "latent_inference_config": asdict(latent_inference_config),
                "fit_seed": fit_seed,
            },
        ),
    )
    def wandb_update(
        update
    ):
        wandb_update = {
            "step": update["opt_step"],
            "elapsed_time_s": update["elapsed_time_s"],
            "loss": update["loss"],
        }

        wandb_run.log(wandb_update)

    run_tracker = train_latent.LatentFitTracker(
        record_trigger=make_record_trigger(10),
        custom_record_fcns=[wandb_update],
    )
    
    fitted_approximation, opt_state, tracker, is_diagnostics = train_latent.train(
        model=latent_approximation,
        embedder=embed,
        dataset=dataset,
        target=target,
        params=params,
        key=train_key,
        optim=optim,
        run_tracker=run_tracker,
        num_steps=latent_inference_config.optimization.total_steps,
        time_limit_s=latent_inference_config.optimization.time_limit_s,
        sample_kwargs=latent_sampling_kwargs,
        unroll=latent_inference_config.unroll,
        compiled_steps=latent_inference_config.compiled_steps,
    )

    sink = WandbLatentFitArtifactSink(wandb_run)
    sink.save(
        run_name=wandb_run.name,
        fitted_approximation=fitted_approximation,
        optimization_state=opt_state,
        tracker_rows=tracker.update_rows,
        is_diagnostics=is_diagnostics,
    )
    wandb_run.finish()


__all__ = ["app", "main", "run", "latent_fit", "list_models", "list_inference", "show_config"]
