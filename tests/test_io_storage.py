import numpy as np

from seqjax.io import LocalFilesystemDataStorage
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
