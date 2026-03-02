import numpy as np
import pytest

from seqjax.io import (
    LocalFilesystemDataStorage,
    LocalPreparedDataStorage,
    dataset_reference_from_data_config,
    save_named_packable_dataset,
)
from seqjax.model.registry import DataConfig


def test_local_filesystem_data_storage_roundtrips_dataset(tmp_path) -> None:
    data_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=32,
        seed=0,
    )
    storage = LocalFilesystemDataStorage(str(tmp_path))

    generated = storage.get_data(data_config)
    loaded = storage.get_data(data_config)

    for generated_item, loaded_item in zip(generated, loaded, strict=True):
        if generated_item is None:
            assert loaded_item is None
            continue
        assert loaded_item is not None
        np.testing.assert_allclose(np.asarray(generated_item.ravel()), np.asarray(loaded_item.ravel()))


def test_local_prepared_data_storage_loads_named_dataset(tmp_path) -> None:
    data_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=32,
        seed=0,
    )
    generating_storage = LocalFilesystemDataStorage(str(tmp_path))
    generated = generating_storage.get_data(data_config)

    save_named_packable_dataset(
        str(tmp_path),
        data_config.dataset_name,
        generated[0],
        generated[1],
        generated[2],
        model_label=data_config.target_model_label,
        sequence_length=data_config.sequence_length,
        num_sequences=data_config.num_sequences,
        overwrite=True,
    )

    loaded = LocalPreparedDataStorage(str(tmp_path)).get_data(
        data_config,
        dataset_reference_from_data_config(data_config),
    )

    for generated_item, loaded_item in zip(generated, loaded, strict=True):
        if generated_item is None:
            assert loaded_item is None
            continue
        assert loaded_item is not None
        np.testing.assert_allclose(np.asarray(generated_item.ravel()), np.asarray(loaded_item.ravel()))


def test_local_prepared_data_storage_raises_if_missing_dataset(tmp_path) -> None:
    data_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=16,
        seed=1,
    )

    with pytest.raises(FileNotFoundError, match="missing required files"):
        LocalPreparedDataStorage(str(tmp_path)).get_data(
            data_config,
            dataset_reference_from_data_config(data_config),
        )


def test_local_prepared_data_storage_raises_on_manifest_mismatch(tmp_path) -> None:
    data_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=16,
        seed=2,
    )
    generating_storage = LocalFilesystemDataStorage(str(tmp_path))
    generated = generating_storage.get_data(data_config)

    save_named_packable_dataset(
        str(tmp_path),
        data_config.dataset_name,
        generated[0],
        generated[1],
        generated[2],
        model_label="ar",
        sequence_length=data_config.sequence_length,
        num_sequences=data_config.num_sequences,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="incompatible"):
        LocalPreparedDataStorage(str(tmp_path)).get_data(
            data_config,
            dataset_reference_from_data_config(data_config),
        )
