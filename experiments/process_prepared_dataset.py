"""Create a prepared local dataset that can be loaded by name.

This script demonstrates the real-data workflow shape:
1. build packables (in real use, parse raw files here),
2. save once under a dataset name,
3. load by name through ``LocalPreparedDataStorage``.
"""

from __future__ import annotations

import argparse
import tempfile

from seqjax import io
from seqjax.model.registry import DataConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Local dataset root (defaults to temp dir)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=16,
        seed=7,
    )

    root = args.root or tempfile.mkdtemp(prefix="seqjax-prepared-")

    # For demonstration we simulate source data. In a real processing script,
    # this is where raw data ingestion/transforms would happen.
    x_path, observation_path, condition = io._simulate_data(data_config)
    io.save_named_packable_dataset(
        root,
        data_config.dataset_name,
        x_path,
        observation_path,
        condition,
        model_label=data_config.target_model_label,
        sequence_length=data_config.sequence_length,
        num_sequences=data_config.num_sequences,
        overwrite=args.overwrite,
    )

    dataset_reference = io.dataset_reference_from_data_config(data_config)
    loaded = io.LocalPreparedDataStorage(root).get_data(data_config, dataset_reference)
    print(f"Prepared dataset saved and loaded: {data_config.dataset_name}")
    print(f"x shape: {loaded[0].ravel().shape}, obs shape: {loaded[1].ravel().shape}")


if __name__ == "__main__":
    main()
