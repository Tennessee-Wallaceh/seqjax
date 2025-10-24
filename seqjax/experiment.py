from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast

import jax.random as jrandom
import wandb

from seqjax import io
from seqjax.inference import registry as inference_registry
from seqjax.model import registry as model_registry


class ResultProcessor(Protocol):
    """Protocol for experiment-specific result processing."""

    def process(
        self,
        run: Any,
        config: "ExperimentConfig",
        param_samples: Any,
        extra_data: Any,
        x_path: Any,
        y_path: Any,
        condition: Any,
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

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ExperimentConfig":
        data_config = model_registry.DataConfig.from_dict(config_dict["data_config"])
        inference = inference_registry.from_dict(config_dict["inference"])
        return cls(
            data_config=data_config,
            test_samples=config_dict["test_samples"],
            fit_seed=config_dict["fit_seed"],
            inference=inference,
        )


def process_results(
    run: Any,
    experiment_config: ExperimentConfig,
    param_samples: Any,
    extra_data: Any,
    x_path: Any,
    y_path: Any,
    condition: Any,
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
        x_path,
        y_path,
        condition,
    )


def run_experiment(
    experiment_name: str,
    experiment_config: ExperimentConfig,
    result_processor: ResultProcessor | None = None,
):
    """Execute an experiment using the shared harness."""

    config_dict = asdict(experiment_config)

    data_wandb_run = cast(io.WandbRun, wandb.init(project=experiment_name))

    target_params = experiment_config.data_config.generative_parameters
    model = experiment_config.posterior_factory(target_params)

    x_path, y_path, condition = io.get_remote_data(
        data_wandb_run, experiment_config.data_config
    )

    data_wandb_run.finish()

    inference = inference_registry.build_inference(experiment_config.inference)

    wandb_run = cast(
        io.WandbRun,
        wandb.init(
            project=experiment_name,
            config={
                **config_dict,
                "inference_name": experiment_config.inference.name,
            },
        ),
    )
    param_samples, extra_data = inference(
        model,
        hyperparameters=None,
        key=jrandom.key(experiment_config.fit_seed),
        observation_path=y_path,
        condition_path=condition,
        test_samples=experiment_config.test_samples,
        config=experiment_config.inference.config,
        wandb_run=wandb_run,
    )

    wandb_run = cast(
        io.WandbRun,
        wandb.init(
            project=experiment_name,
            config={
                **config_dict,
                "inference_name": experiment_config.inference.name,
            },
        ),
    )
    process_results(
        wandb_run,
        experiment_config,
        param_samples,
        extra_data,
        x_path,
        y_path,
        condition,
        result_processor,
    )
    wandb_run.finish()

    return (param_samples, extra_data, x_path, y_path)
