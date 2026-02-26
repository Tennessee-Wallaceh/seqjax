from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Literal, Protocol, cast

import jax.random as jrandom
import wandb

from seqjax import io
from seqjax.inference import interface as inference_interface
from seqjax.inference import registry as inference_registry
from seqjax.model import registry as model_registry
from seqjax.inference import vi
import seqjax.model.typing as seqjtyping


def make_record_trigger(interval_seconds: int):
    last_trigger = [-1]

    def trigger(step, elapsed_time):
        current = int(elapsed_time) // interval_seconds
        if current != last_trigger[0]:
            last_trigger[0] = current
            return True
        return False

    return trigger


class ResultProcessor(Protocol):
    """Protocol for experiment-specific result processing."""

    def process(
        self,
        run: Any,
        config: "ExperimentConfig",
        param_samples: Any,
        extra_data: Any,
        x_paths: Any,
        observation_paths: Any,
        conditions: Any,
    ) -> None: ...


@dataclass
class ExperimentConfig:
    """Configuration for running an experiment through :func:`run_experiment`."""

    data_config: model_registry.DataConfig
    test_samples: int
    fit_seed: int
    inference: inference_registry.InferenceConfig

    @property
    def posterior_factory(self) -> model_registry.PosteriorFactory:
        return self.data_config.posterior_factory


StorageMode = Literal["wandb", "wandb-offline"]


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime-only settings for experiment execution and tracking."""

    storage_mode: StorageMode = "wandb"
    local_root: str = "./wandb"

    @property
    def wandb_offline(self) -> bool:
        return self.storage_mode == "wandb-offline"


def configure_wandb_runtime(runtime_config: RuntimeConfig) -> None:
    """Set environment variables expected by W&B for local/offline execution."""

    if runtime_config.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = runtime_config.local_root
    else:
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DIR", None)



def process_results(
    run: Any,
    experiment_config: ExperimentConfig,
    param_samples: Any,
    extra_data: Any,
    x_paths: Any,
    observation_paths: Any,
    conditions: Any,
    result_processor: ResultProcessor | None,
) -> None:
    """Delegate experiment results to a result processor if provided."""

    if result_processor is None:
        return

    result_processor.process(
        run,
        experiment_config,
        param_samples,
        extra_data,
        x_paths,
        observation_paths,
        conditions,
    )


def build_tracker(experiment_config: ExperimentConfig, wandb_run):
    run_tracker = None

    if (
        experiment_config.inference.label == "buffer-vi"
        or experiment_config.inference.label == "full-vi"
    ):

        def wandb_update(
            update, static, trainable, opt_step, loss, loss_label, key
        ):
            wandb_update = {
                "step": opt_step,
                "elapsed_time_s": update["elapsed_time_s"],
                "loss": loss,
                loss_label: float(loss),
            }

            for label, value in update.items():
                if label.endswith("_q05"):
                    wandb_update[label] = value
                if label.endswith("_q95"):
                    wandb_update[label] = value
                if label.endswith("_mean"):
                    wandb_update[label] = value

            wandb_run.log(wandb_update)

        custom_record_fcns = [wandb_update]

        run_tracker = vi.train.Tracker(
            record_trigger=make_record_trigger(30),
            metric_samples=experiment_config.test_samples,
            custom_record_fcns=custom_record_fcns,
        )

    return run_tracker


def run_experiment(
    experiment_name: str,
    experiment_config: ExperimentConfig,
    result_processor: ResultProcessor | None = None,
    runtime_config: RuntimeConfig | None = None,
):
    """Execute an experiment using the shared harness."""

    resolved_runtime_config = runtime_config or RuntimeConfig()
    configure_wandb_runtime(resolved_runtime_config)

    config_dict = asdict(experiment_config)

    data_wandb_run = cast(io.WandbRun, wandb.init(project=experiment_name))

    target_params = experiment_config.data_config.generative_parameters
    model = experiment_config.posterior_factory(target_params)

    
    data_storage: io.DataStorage
    if resolved_runtime_config.wandb_offline:
        data_storage = io.LocalFilesystemDataStorage(resolved_runtime_config.local_root)
    else:
        data_storage = io.WandbArtifactDataStorage(data_wandb_run)

    x_paths, observations, conditions = data_storage.get_remote_data(
        data_wandb_run, experiment_config.data_config
    )
    condition_paths = seqjtyping.NoCondition() if conditions is None else conditions

    dataset = inference_interface.ObservationDataset(
        observations=observations,
        conditions=condition_paths,
    )

    data_wandb_run.finish()

    inference = experiment_config.inference.run
    wandb_run = cast(
        io.WandbRun,
        wandb.init(
            project=experiment_name,
            config={
                **config_dict,
                "inference_name": experiment_config.inference.name,
                "results": False,
            },
        ),
    )
    param_samples, extra_data = inference(
        model,
        hyperparameters=None,
        key=jrandom.key(experiment_config.fit_seed),
        dataset=dataset,
        test_samples=experiment_config.test_samples,
        config=experiment_config.inference.config,
        tracker=build_tracker(experiment_config, wandb_run),
    )

    wandb_run.finish()

    process_wandb_run = cast(
        io.WandbRun,
        wandb.init(
            project=experiment_name,
            config={
                **config_dict,
                "inference_name": experiment_config.inference.name,
                "training_run_id": wandb_run.id,
                "training_run_name": wandb_run.name,
                "results": True,
            },
            settings=wandb.Settings(start_method="thread"),
        ),
    )
    process_results(
        process_wandb_run,
        experiment_config,
        param_samples,
        extra_data,
        x_paths,
        observations,
        conditions,
        result_processor,
    )
    process_wandb_run.finish()

    return (param_samples, extra_data, x_paths, observations)
