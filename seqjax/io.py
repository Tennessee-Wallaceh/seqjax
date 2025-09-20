from collections.abc import Mapping
from typing import Any, Protocol

import polars as pl  # type: ignore[import-not-found]
import numpy as np
import os
from seqjax.model.typing import Packable
from seqjax.model.registry import DataConfig
from seqjax.model import simulate
import jax.random as jrandom

import wandb
import wandb.errors


class WandbRun(Protocol):
    """Subset of the :mod:`wandb` run API used by this module."""

    def log_artifact(self, artifact: wandb.Artifact) -> None: ...

    def use_artifact(self, artifact_name: str) -> wandb.Artifact: ...

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


def packable_artifact_present(
    run: WandbRun, artifact_name: str, file_name: str | None = None
):
    try:
        artifact = run.use_artifact(f"{artifact_name}:latest")
    except wandb.errors.CommError:
        return False

    if file_name is not None:
        artifact_dir = artifact.download()

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
    artifact_dir = artifact.download()
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
    artifact_dir = artifact.download()
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

    if not packable_artifact_present(run, artifact_name):
        print(f"{artifact_name} not present on remote, generating...")
        data_key = jrandom.PRNGKey(data_config.seed)
        x_path, y_path, _, _ = simulate.simulate(
            data_key,
            data_config.target,
            None,
            data_config.generative_parameters,  # needs params
            sequence_length=data_config.sequence_length,
        )

        print(f"saving {artifact_name} on remote...")
        save_packable_artifact(
            run,
            artifact_name,
            "dataset",
            [
                ("x_path", x_path, {}),
                ("y_path", y_path, {}),
            ],
        )
        return x_path, y_path

    print(f"{artifact_name} present on remote, downloading...")
    (x_path, _), (y_path, _) = load_packable_artifact(
        run,
        artifact_name,
        [
            ("x_path", data_config.target.particle_cls),
            ("y_path", data_config.target.observation_cls),
        ],
    )

    return x_path, y_path
