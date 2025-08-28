import polars as pl
import numpy as np
from seqjax.model.typing import Packable
import wandb
import os

SEQJAX_DATA_DIR = "../"


def packable_to_df(packable: Packable) -> pl.DataFrame:
    flat_array = np.asarray(packable.ravel(packable))
    return pl.DataFrame(flat_array, schema=packable.flat_fields())


def df_to_packable(packable_cls: type[Packable], df: pl.DataFrame) -> Packable:
    return packable_cls.unravel(df.to_jax())


def save_packable_artifact(
    run: wandb.Run,
    artifact_name: str,
    file_name: str,
    wandb_type: str,
    packable: Packable,
    metadata: dict,
):
    file_loc = f"{SEQJAX_DATA_DIR}/{file_name}.parquet"
    artifact = wandb.Artifact(name=artifact_name, type=wandb_type)

    df = packable_to_df(packable)
    df.write_parquet(file_loc, metadata=metadata)
    artifact.add_file(local_path=file_loc)
    run.log_artifact(artifact)


def packable_artifact_present(
    run: wandb.Run, artifact_name: str, file_name: str | None = None
):
    try:
        artifact = run.use_artifact(f"{artifact_name}:latest")
    except wandb.CommError:
        return False

    if file_name is not None:
        artifact_dir = artifact.download()

        file_path = os.path.join(artifact_dir, f"{file_name}.parquet")

        if not os.path.isfile(file_path):
            return False

    return True


def load_packable_artifact(
    run: wandb.Run,
    artifact_name: str,
    file_name: str,
    packable_cls: type[Packable],
) -> tuple[Packable, dict]:
    artifact = run.use_artifact(f"{artifact_name}:latest")
    artifact_dir = artifact.download()
    file_path = os.path.join(artifact_dir, f"{file_name}.parquet")
    return (
        df_to_packable(packable_cls, pl.read_parquet(file_path)),
        pl.read_parquet_metadata(file_path),
    )
