from collections.abc import Mapping
from typing import Any, Protocol
import typing

import equinox as eqx

import polars as pl  # type: ignore[import-not-found]
import numpy as np
import os
from seqjax.model.typing import Packable
from seqjax.model.registry import DataConfig, condition_generators
from seqjax.model import simulate
import jax.random as jrandom

import wandb
import wandb.errors


class WandbRun(Protocol):
    """Subset of the :mod:`wandb` run API used by this module."""

    name: str
    project: str
    id: str

    def log_artifact(self, artifact: wandb.Artifact) -> None: ...

    def use_artifact(self, artifact_name: str) -> wandb.Artifact: ...

    def finish(self) -> None: ...


SEQJAX_DATA_DIR = "../"


def normalize_parquet_metadata(md: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in md.items():
        if k == "elapsed_time_s":
            # this is not brilliant encoding, but the quantity does not
            # need massive precision
            out[k] = str(v)
        else:
            out[k] = v
    return out


def process_parquet_metadata(md: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in md.items():
        if k == "elapsed_time_s":
            out[k] = float(v)
        else:
            out[k] = v
    return out


def packable_to_df(packable: Packable) -> pl.DataFrame:
    flat_array = np.asarray(packable.ravel(packable))
    return pl.DataFrame(flat_array, schema=packable.flat_fields())


def df_to_packable(packable_cls: type[Packable], df: pl.DataFrame) -> Packable:
    return packable_cls.unravel(df.to_jax())


def download_artifact(artifact: wandb.Artifact) -> str:
    artifact_dir = artifact.download()
    if os.path.isdir(artifact_dir) is False:
        if os.path.isdir(artifact_dir.replace(":", "-")) is False:
            raise ValueError(
                f"could not locate {artifact_dir} or {artifact_dir.replace(':', '-')}!"
            )
        else:
            artifact_dir = artifact_dir.replace(":", "-")
    return artifact_dir


def save_packable_artifact(
    run: WandbRun,
    artifact_name: str,
    wandb_type: str,
    file_names_and_data: list[tuple[str, Packable, dict]],
):
    artifact = wandb.Artifact(name=artifact_name, type=wandb_type)

    for file_name, packable, metadata in file_names_and_data:
        file_loc = f"{SEQJAX_DATA_DIR}/{file_name}.parquet"
        df = packable_to_df(packable)
        df.write_parquet(file_loc, metadata=normalize_parquet_metadata(metadata))
        artifact.add_file(local_path=file_loc)
    run.log_artifact(artifact)


def save_model_artifact(
    run: WandbRun,
    artifact_name: str,
    model: eqx.Module,
):
    artifact = wandb.Artifact(name=artifact_name, type="approximation")
    file_loc = f"{SEQJAX_DATA_DIR}/{artifact_name}.eqx"

    with open(file_loc, "wb") as f:
        eqx.tree_serialise_leaves(f, model)

    artifact.add_file(local_path=file_loc)
    run.log_artifact(artifact)


def load_model_artifact(
    target_run: WandbRun,
    artifact_name: str,
    model: eqx.Module,
) -> eqx.Module:
    print(
        f"Loading {target_run.name}-{artifact_name}:latest from wandb {target_run.project}..."
    )
    run = wandb.init(project=target_run.project)
    artifact = run.use_artifact(
        f"{target_run.name}-{artifact_name}:latest", type="approximation"
    )

    artifact_dir = download_artifact(artifact)
    file_path = os.path.join(artifact_dir, f"{target_run.name}-{artifact_name}.eqx")

    with open(file_path, "rb") as f:
        model = eqx.tree_deserialise_leaves(f, like=model)

    return model


def packable_artifact_present(
    run: WandbRun, artifact_name: str, file_name: str | None = None
):
    try:
        artifact = run.use_artifact(f"{artifact_name}:latest")
    except wandb.errors.CommError:
        return False

    if file_name is not None:
        artifact_dir = download_artifact(artifact)

        file_path = os.path.join(artifact_dir, f"{file_name}.parquet")

        if not os.path.isfile(file_path):
            return False

    return True


def load_packable_artifact(
    run: WandbRun,
    artifact_name: str,
    file_names_and_class: list[tuple[str, type[Packable]]],
) -> list[tuple[Packable, dict]]:
    artifact = run.use_artifact(f"{artifact_name}:latest")
    artifact_dir = download_artifact(artifact)
    loaded_data = []
    for file_name, packable_cls in file_names_and_class:
        file_path = os.path.join(artifact_dir, f"{file_name}.parquet")
        loaded_data.append(
            (
                df_to_packable(packable_cls, pl.read_parquet(file_path)),
                process_parquet_metadata(pl.read_parquet_metadata(file_path)),
            )
        )
    return loaded_data


def load_packable_artifact_all(
    run: WandbRun,
    artifact_name: str,
    packable_cls: type[Packable],
) -> list[tuple[Packable, dict]]:
    artifact = run.use_artifact(f"{artifact_name}:latest")
    artifact_dir = download_artifact(artifact)

    loaded_data = []
    files = [e.name for e in os.scandir(artifact_dir) if e.is_file()]
    for file_name in files:
        if file_name.endswith(".parquet"):
            file_path = os.path.join(artifact_dir, file_name)
            loaded_data.append(
                (
                    df_to_packable(packable_cls, pl.read_parquet(file_path)),
                    process_parquet_metadata(pl.read_parquet_metadata(file_path)),
                )
            )
    return loaded_data


def get_remote_data(run: WandbRun, data_config: DataConfig):
    artifact_name = data_config.dataset_name
    condition = None

    if not packable_artifact_present(run, artifact_name):
        print(f"{artifact_name} not present on remote, generating...")
        data_key = jrandom.PRNGKey(data_config.seed)

        if data_config.target_model_label in condition_generators:
            condition = condition_generators[data_config.target_model_label](
                data_config.sequence_length
            )

        x_path, y_path = simulate.simulate(
            data_key,
            data_config.target,
            data_config.generative_parameters,  # needs params
            sequence_length=data_config.sequence_length,
            condition=condition,
        )

        print(f"saving {artifact_name} on remote...")
        to_save: typing.Any = [
            ("x_path", x_path, {}),
            ("y_path", y_path, {}),
        ]
        if condition is not None:
            to_save.append(
                ("condition", condition, {}),
            )
        save_packable_artifact(run, artifact_name, "dataset", to_save)
        return x_path, y_path, condition

    print(f"{artifact_name} present on remote, downloading...")
    to_load = [
        ("x_path", data_config.target.latent_cls),
        ("y_path", data_config.target.observation_cls),
    ]
    if data_config.target_model_label in condition_generators:
        to_load.append(("condition", data_config.target.condition_cls))

    loaded = load_packable_artifact(run, artifact_name, to_load)

    if data_config.target_model_label in condition_generators:
        (x_path, _), (y_path, _), (condition, _) = loaded
    else:
        (x_path, _), (y_path, _) = loaded

    return x_path, y_path, condition
