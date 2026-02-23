import os

from seqjax.experiment import RuntimeConfig, configure_wandb_runtime


def test_configure_wandb_runtime_offline_sets_environment() -> None:
    os.environ.pop("WANDB_MODE", None)
    os.environ.pop("WANDB_DIR", None)

    config = RuntimeConfig(storage_mode="wandb-offline", local_root="/tmp/seqjax-local")
    configure_wandb_runtime(config)

    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["WANDB_DIR"] == "/tmp/seqjax-local"


def test_configure_wandb_runtime_online_clears_environment() -> None:
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/tmp/seqjax-local"

    config = RuntimeConfig(storage_mode="wandb", local_root="/tmp/ignored")
    configure_wandb_runtime(config)

    assert "WANDB_MODE" not in os.environ
    assert "WANDB_DIR" not in os.environ
