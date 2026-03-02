from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
import typing
import json
import pickle

import equinox as eqx

import polars as pl  # type: ignore[import-not-found]
import numpy as np
import os
from seqjax.model.typing import Packable
from seqjax.model.registry import DataConfig, condition_generators
from seqjax.model import simulate
import jax.random as jrandom
import jax
import jax.numpy as jnp

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


class DatasetReference(Protocol):
    """Reference contract used to locate and validate a prepared dataset."""

    dataset_name: str
    expected_model_label: str | None
    expected_sequence_length: int | None
    expected_num_sequences: int | None


@dataclass
class NamedDatasetReference:
    dataset_name: str
    expected_model_label: str | None = None
    expected_sequence_length: int | None = None
    expected_num_sequences: int | None = None


def dataset_reference_from_data_config(
    data_config: DataConfig,
    dataset_name_override: str | None = None,
) -> NamedDatasetReference:
    return NamedDatasetReference(
        dataset_name=data_config.dataset_name if dataset_name_override is None else dataset_name_override,
        expected_model_label=data_config.target_model_label,
        expected_sequence_length=data_config.sequence_length,
        expected_num_sequences=data_config.num_sequences,
    )


class PreparedDataStorage(Protocol):
    """Storage backend interface for pre-processed, named datasets."""

    def get_data(
        self,
        data_config: DataConfig,
        dataset_reference: DatasetReference,
    ) -> tuple[Packable, Packable, Packable | None]: ...


SEQJAX_DATA_DIR = "../"


def _stack_packables(packables: list[Packable]) -> Packable:
    if len(packables) == 0:
        raise ValueError("Cannot stack an empty packable list.")
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves, axis=0),
        *packables,
    )


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


def _read_packable_parquet(file_path: str, packable_cls: type[Packable]) -> Packable:
    expected_columns = list(packable_cls.flat_fields())
    df = pl.read_parquet(file_path)
    actual_columns = df.columns

    if set(actual_columns) != set(expected_columns):
        missing = sorted(set(expected_columns) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(expected_columns))
        raise ValueError(
            "Packable parquet schema mismatch for "
            f"{file_path}. Missing={missing}, extra={extra}."
        )

    if actual_columns != expected_columns:
        df = df.select(expected_columns)

    return df_to_packable(packable_cls, df)




PREPARED_DATASET_MANIFEST = "dataset_manifest.json"


def _manifest_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, PREPARED_DATASET_MANIFEST)


def _write_dataset_manifest(
    dataset_dir: str,
    dataset_reference: NamedDatasetReference,
) -> None:
    payload = {
        "dataset_name": dataset_reference.dataset_name,
        "model_label": dataset_reference.expected_model_label,
        "sequence_length": dataset_reference.expected_sequence_length,
        "num_sequences": dataset_reference.expected_num_sequences,
    }
    with open(_manifest_path(dataset_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_dataset_manifest(dataset_dir: str) -> dict[str, Any]:
    manifest_path = _manifest_path(dataset_dir)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Prepared dataset manifest not found: {manifest_path}. "
            "Re-process data with save_named_packable_dataset to include safety metadata."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return typing.cast(dict[str, Any], json.load(f))


def _validate_dataset_manifest(
    dataset_dir: str,
    dataset_reference: DatasetReference,
) -> None:
    manifest = _read_dataset_manifest(dataset_dir)

    if manifest.get("dataset_name") != dataset_reference.dataset_name:
        raise ValueError(
            "Prepared dataset manifest mismatch: "
            f"expected dataset_name={dataset_reference.dataset_name}, "
            f"found {manifest.get('dataset_name')}."
        )

    checks = [
        ("model_label", dataset_reference.expected_model_label),
        ("sequence_length", dataset_reference.expected_sequence_length),
        ("num_sequences", dataset_reference.expected_num_sequences),
    ]
    for field_name, expected in checks:
        if expected is None:
            continue
        if manifest.get(field_name) != expected:
            raise ValueError(
                "Prepared dataset is incompatible with requested configuration: "
                f"{field_name} expected {expected}, found {manifest.get(field_name)}."
            )

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

def _get_remote_data_legacy(run: WandbRun, data_config: DataConfig):
    artifact_name = data_config.dataset_name

    if not packable_artifact_present(run, artifact_name):
        print(f"{artifact_name} not present on remote, generating...")
        data_key = jrandom.PRNGKey(data_config.seed)
        sequence_keys = jrandom.split(data_key, data_config.num_sequences)

        x_paths: list[Packable] = []
        observation_paths: list[Packable] = []
        condition_paths: list[Packable | None] = []

        for sequence_idx, sequence_key in enumerate(sequence_keys):
            condition = None
            if data_config.target_model_label in condition_generators:
                condition = condition_generators[data_config.target_model_label](
                    data_config.sequence_length
                )

            x_path, observation_path = simulate.simulate(
                sequence_key,
                data_config.target,
                data_config.generative_parameters,
                sequence_length=data_config.sequence_length,
                condition=condition,
            )
            x_paths.append(x_path)
            observation_paths.append(observation_path)
            condition_paths.append(condition)

        print(f"saving {artifact_name} on remote...")
        to_save: typing.Any = []
        if data_config.num_sequences == 1:
            to_save.extend([
                ("x_path", x_paths[0], {}),
                ("observation_path", observation_paths[0], {}),
            ])
            if condition_paths[0] is not None:
                to_save.append(("condition", condition_paths[0], {}))
        else:
            for sequence_idx, (x_path, observation_path) in enumerate(zip(x_paths, observation_paths)):
                to_save.extend([
                    (f"x_path_s{sequence_idx}", x_path, {"sequence_idx": sequence_idx}),
                    (f"observation_path_s{sequence_idx}", observation_path, {"sequence_idx": sequence_idx}),
                ])
                condition = condition_paths[sequence_idx]
                if condition is not None:
                    to_save.append((f"condition_s{sequence_idx}", condition, {"sequence_idx": sequence_idx}))

        save_packable_artifact(run, artifact_name, "dataset", to_save)

        x_stacked = _stack_packables(x_paths)
        observation_stacked = _stack_packables(observation_paths)
        if all(c is not None for c in condition_paths):
            condition_stacked = _stack_packables(typing.cast(list[Packable], condition_paths))
        else:
            condition_stacked = None
        return x_stacked, observation_stacked, condition_stacked

    print(f"{artifact_name} present on remote, downloading...")
    if data_config.num_sequences == 1:
        to_load = [
            ("x_path", data_config.target.latent_cls),
            ("observation_path", data_config.target.observation_cls),
        ]
        if data_config.target_model_label in condition_generators:
            to_load.append(("condition", data_config.target.condition_cls))

        loaded = load_packable_artifact(run, artifact_name, to_load)

        if data_config.target_model_label in condition_generators:
            (x_path, _), (observation_path, _), (condition, _) = loaded
        else:
            (x_path, _), (observation_path, _) = loaded
            condition = None

        x_stacked = _stack_packables([x_path])
        observation_stacked = _stack_packables([observation_path])
        condition_stacked = _stack_packables(typing.cast(list[Packable], [condition])) if condition is not None else None
        return x_stacked, observation_stacked, condition_stacked

    to_load = []
    for sequence_idx in range(data_config.num_sequences):
        to_load.extend([
            (f"x_path_s{sequence_idx}", data_config.target.latent_cls),
            (f"observation_path_s{sequence_idx}", data_config.target.observation_cls),
        ])
        if data_config.target_model_label in condition_generators:
            to_load.append((f"condition_s{sequence_idx}", data_config.target.condition_cls))

    loaded = load_packable_artifact(run, artifact_name, to_load)

    x_paths = []
    observation_paths = []
    condition_paths = []
    idx = 0
    for _ in range(data_config.num_sequences):
        (x_path, _), (observation_path, _) = loaded[idx], loaded[idx + 1]
        idx += 2
        x_paths.append(x_path)
        observation_paths.append(observation_path)
        if data_config.target_model_label in condition_generators:
            (condition, _) = loaded[idx]
            idx += 1
            condition_paths.append(condition)

    x_stacked = _stack_packables(x_paths)
    observation_stacked = _stack_packables(observation_paths)
    condition_stacked = _stack_packables(typing.cast(list[Packable], condition_paths)) if condition_paths else None
    return x_stacked, observation_stacked, condition_stacked

  
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


def save_named_packable_dataset(
    local_root: str,
    dataset_name: str,
    x_path: Packable,
    observation_path: Packable,
    condition: Packable | None,
    *,
    model_label: str | None = None,
    sequence_length: int | None = None,
    num_sequences: int | None = None,
    overwrite: bool = False,
) -> None:
    """Persist a pre-processed dataset under ``local_root/datasets/<dataset_name>``."""

    dataset_dir = os.path.join(local_root, "datasets", dataset_name)
    if os.path.isdir(dataset_dir) and not overwrite:
        raise FileExistsError(
            f"Dataset directory already exists: {dataset_dir}. "
            "Pass overwrite=True to replace it."
        )

    os.makedirs(dataset_dir, exist_ok=True)
    packable_to_df(x_path).write_parquet(os.path.join(dataset_dir, "x_path.parquet"))
    packable_to_df(observation_path).write_parquet(
        os.path.join(dataset_dir, "observation_path.parquet")
    )
    if condition is not None:
        packable_to_df(condition).write_parquet(
            os.path.join(dataset_dir, "condition.parquet")
        )

    _write_dataset_manifest(
        dataset_dir,
        NamedDatasetReference(
            dataset_name=dataset_name,
            expected_model_label=model_label,
            expected_sequence_length=sequence_length,
            expected_num_sequences=num_sequences,
        ),
    )


class LocalPreparedDataStorage:
    """Named dataset backend that only loads pre-processed local datasets."""

    def __init__(self, local_root: str):
        self._base_dir = os.path.join(local_root, "datasets")

    def _dataset_dir(self, dataset_name: str) -> str:
        return os.path.join(self._base_dir, dataset_name)

    def _file_path(self, dataset_name: str, file_name: str) -> str:
        return os.path.join(self._dataset_dir(dataset_name), f"{file_name}.parquet")

    def _validate_dataset_files(self, dataset_name: str, data_config: DataConfig) -> None:
        required = ["x_path", "observation_path"]
        if data_config.target_model_label in condition_generators:
            required.append("condition")

        missing = [
            name
            for name in required
            if not os.path.isfile(self._file_path(dataset_name, name))
        ]
        if missing:
            raise FileNotFoundError(
                f"Prepared dataset '{dataset_name}' is missing required files: {missing}. "
                f"Expected directory: {self._dataset_dir(dataset_name)}"
            )

    def get_data(
        self,
        data_config: DataConfig,
        dataset_reference: DatasetReference,
    ) -> tuple[Packable, Packable, Packable | None]:
        dataset_name = dataset_reference.dataset_name
        self._validate_dataset_files(dataset_name, data_config)
        _validate_dataset_manifest(self._dataset_dir(dataset_name), dataset_reference)

        x_path = _read_packable_parquet(
            self._file_path(dataset_name, "x_path"),
            data_config.target.latent_cls,
        )
        observation_path = _read_packable_parquet(
            self._file_path(dataset_name, "observation_path"),
            data_config.target.observation_cls,
        )

        condition: Packable | None = None
        if data_config.target_model_label in condition_generators:
            condition = _read_packable_parquet(
                self._file_path(dataset_name, "condition"),
                data_config.target.condition_cls,
            )

        return x_path, observation_path, condition


def get_remote_data(run: WandbRun, data_config: DataConfig):
    return WandbArtifactDataStorage(run).get_data(data_config)
