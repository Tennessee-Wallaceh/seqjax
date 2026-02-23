from collections.abc import Mapping
from typing import Any, Protocol
import typing
import pickle

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


class DataStorage(Protocol):
    """Storage backend interface used to load or create datasets."""

    def get_data(self, data_config: DataConfig) -> tuple[Packable, Packable, Packable | None]: ...


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
    flat_array = np.asarray(packable.ravel())
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


def save_python_artifact(
    run: WandbRun,
    artifact_name: str,
    wandb_type: str,
    file_names_and_data: list[tuple[str, object]],
):
    artifact = wandb.Artifact(name=artifact_name, type=wandb_type)

    for file_name, python_object in file_names_and_data:
        file_loc = f"{SEQJAX_DATA_DIR}/{file_name}.pkl"
        with open(file_loc, 'wb') as f:
            pickle.dump(python_object, f)
        artifact.add_file(local_path=file_loc)
    run.log_artifact(artifact)


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

def load_python_object(
    target_run: WandbRun,
    artifact_name: str,
    file_name: str,
) -> eqx.Module:
    print(
        f"Loading {target_run.name}-{artifact_name}:latest from wandb {target_run.project}..."
    )
    run = wandb.init(project=target_run.project)
    artifact = run.use_artifact(
        f"{target_run.name}-{artifact_name}:latest", type="run_output"
    )

    artifact_dir = download_artifact(artifact)
    file_path = os.path.join(artifact_dir, f"{file_name}.pkl")

    with open(file_path, "rb") as f:
        python_object = pickle.load(f)

    return python_object


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


def _simulate_data(data_config: DataConfig) -> tuple[Packable, Packable, Packable | None]:
    condition: Packable | None = None
    data_key = jrandom.PRNGKey(data_config.seed)

    if data_config.target_model_label in condition_generators:
        condition = condition_generators[data_config.target_model_label](
            data_config.sequence_length
        )

    x_path, observation_path = simulate.simulate(
        data_key,
        data_config.target,
        data_config.generative_parameters,
        sequence_length=data_config.sequence_length,
        condition=condition,
    )

    return x_path, observation_path, condition


class WandbArtifactDataStorage:
    """Dataset storage backend backed by W&B artifacts."""

    def __init__(self, run: WandbRun):
        self._run = run

    def get_data(
        self,
        data_config: DataConfig,
    ) -> tuple[Packable, Packable, Packable | None]:
        artifact_name = data_config.dataset_name
        condition: Packable | None = None

        if not packable_artifact_present(self._run, artifact_name):
            print(f"{artifact_name} not present on remote, generating...")
            x_path, observation_path, condition = _simulate_data(data_config)

            print(f"saving {artifact_name} on remote...")
            to_save: typing.Any = [
                ("x_path", x_path, {}),
                ("observation_path", observation_path, {}),
            ]
            if condition is not None:
                to_save.append(("condition", condition, {}))
            save_packable_artifact(self._run, artifact_name, "dataset", to_save)
            return x_path, observation_path, condition

        print(f"{artifact_name} present on remote, downloading...")
        to_load = [
            ("x_path", data_config.target.latent_cls),
            ("observation_path", data_config.target.observation_cls),
        ]
        if data_config.target_model_label in condition_generators:
            to_load.append(("condition", data_config.target.condition_cls))

        loaded = load_packable_artifact(self._run, artifact_name, to_load)

        if data_config.target_model_label in condition_generators:
            (x_path, _), (observation_path, _), (condition, _) = loaded
        else:
            (x_path, _), (observation_path, _) = loaded

        return x_path, observation_path, condition


class LocalFilesystemDataStorage:
    """Dataset storage backend that persists parquet files locally."""

    def __init__(self, local_root: str):
        self._base_dir = os.path.join(local_root, "datasets")

    def _dataset_dir(self, artifact_name: str) -> str:
        return os.path.join(self._base_dir, artifact_name)

    def _file_path(self, artifact_name: str, file_name: str) -> str:
        return os.path.join(self._dataset_dir(artifact_name), f"{file_name}.parquet")

    def _dataset_present(self, data_config: DataConfig) -> bool:
        artifact_name = data_config.dataset_name
        required = ["x_path", "observation_path"]
        if data_config.target_model_label in condition_generators:
            required.append("condition")

        return all(os.path.isfile(self._file_path(artifact_name, name)) for name in required)

    def _save_data(
        self,
        artifact_name: str,
        x_path: Packable,
        observation_path: Packable,
        condition: Packable | None,
    ) -> None:
        dataset_dir = self._dataset_dir(artifact_name)
        os.makedirs(dataset_dir, exist_ok=True)
        packable_to_df(x_path).write_parquet(self._file_path(artifact_name, "x_path"))
        packable_to_df(observation_path).write_parquet(
            self._file_path(artifact_name, "observation_path")
        )
        if condition is not None:
            packable_to_df(condition).write_parquet(
                self._file_path(artifact_name, "condition")
            )

    def _load_data(
        self, data_config: DataConfig
    ) -> tuple[Packable, Packable, Packable | None]:
        artifact_name = data_config.dataset_name
        x_path = df_to_packable(
            data_config.target.latent_cls,
            pl.read_parquet(self._file_path(artifact_name, "x_path")),
        )
        observation_path = df_to_packable(
            data_config.target.observation_cls,
            pl.read_parquet(self._file_path(artifact_name, "observation_path")),
        )

        condition: Packable | None = None
        if data_config.target_model_label in condition_generators:
            condition = df_to_packable(
                data_config.target.condition_cls,
                pl.read_parquet(self._file_path(artifact_name, "condition")),
            )
        return x_path, observation_path, condition

    def get_data(
        self,
        data_config: DataConfig,
    ) -> tuple[Packable, Packable, Packable | None]:
        artifact_name = data_config.dataset_name

        if not self._dataset_present(data_config):
            print(f"{artifact_name} not present locally, generating...")
            x_path, observation_path, condition = _simulate_data(data_config)
            print(f"saving {artifact_name} locally...")
            self._save_data(artifact_name, x_path, observation_path, condition)
            return x_path, observation_path, condition

        print(f"{artifact_name} present locally, loading...")
        return self._load_data(data_config)


def get_remote_data(run: WandbRun, data_config: DataConfig):
    return WandbArtifactDataStorage(run).get_data(data_config)
