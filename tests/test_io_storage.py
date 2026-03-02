import numpy as np
import pytest

from seqjax.io import (
    LocalFilesystemDataStorage,
    LocalPreparedDataStorage,
    NamedPreparedDataRequest,
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


def test_local_filesystem_data_storage_roundtrips_multi_sequence_dataset(tmp_path) -> None:
    data_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=16,
        seed=0,
        num_sequences=3,
    )
    storage = LocalFilesystemDataStorage(str(tmp_path))

    generated = storage.get_data(data_config)
    loaded = storage.get_data(data_config)

    assert generated[1].batch_shape[0] == 3
    assert loaded[1].batch_shape[0] == 3
    for generated_item, loaded_item in zip(generated, loaded, strict=True):
        if generated_item is None:
            assert loaded_item is None
            continue
        assert loaded_item is not None
        np.testing.assert_allclose(np.asarray(generated_item.ravel()), np.asarray(loaded_item.ravel()))


def test_local_filesystem_data_storage_raises_on_partial_multi_sequence_shards(tmp_path) -> None:
    data_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=10,
        seed=5,
        num_sequences=2,
    )
    storage = LocalFilesystemDataStorage(str(tmp_path))
    generated = storage.get_data(data_config)
    assert generated[1].batch_shape[0] == 2

    dataset_dir = tmp_path / "datasets" / data_config.dataset_name
    shard_to_remove = dataset_dir / "observation_path_s1.parquet"
    shard_to_remove.unlink()

    with pytest.raises(FileNotFoundError, match="strict shard layout"):
        storage.get_data(data_config)


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
        NamedPreparedDataRequest(
            dataset_name=data_config.dataset_name,
            target_model_label=data_config.target_model_label,
        ),
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
            NamedPreparedDataRequest(
                dataset_name=data_config.dataset_name,
                target_model_label=data_config.target_model_label,
            ),
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
            NamedPreparedDataRequest(
                dataset_name=data_config.dataset_name,
                target_model_label=data_config.target_model_label,
            ),
        )


def test_local_prepared_data_storage_does_not_require_matching_sequence_length(tmp_path) -> None:
    write_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=16,
        seed=3,
    )
    generating_storage = LocalFilesystemDataStorage(str(tmp_path))
    generated = generating_storage.get_data(write_config)

    save_named_packable_dataset(
        str(tmp_path),
        write_config.dataset_name,
        generated[0],
        generated[1],
        generated[2],
        model_label=write_config.target_model_label,
        sequence_length=write_config.sequence_length,
        num_sequences=write_config.num_sequences,
        overwrite=True,
    )

    read_config = DataConfig(
        target_model_label="aicher_stochastic_vol",
        generative_parameter_label="base",
        sequence_length=32,
        seed=999,
    )
    loaded = LocalPreparedDataStorage(str(tmp_path)).get_data(
        read_config,
        NamedPreparedDataRequest(
            dataset_name=write_config.dataset_name,
            target_model_label=read_config.target_model_label,
        ),
    )

    for generated_item, loaded_item in zip(generated, loaded, strict=True):
        if generated_item is None:
            assert loaded_item is None
            continue
        assert loaded_item is not None
        np.testing.assert_allclose(np.asarray(generated_item.ravel()), np.asarray(loaded_item.ravel()))
