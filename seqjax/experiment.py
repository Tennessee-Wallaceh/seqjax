from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from typing import Any, Literal, Protocol, cast

import jax
import jax.numpy as jnp
import jax.random as jrandom
import wandb
import numpy as np


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
    init_from_generative: bool = False


WandBStorageMode = Literal["wandb", "wandb-offline"]


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime-only settings for experiment execution and tracking."""
    storage_mode: WandBStorageMode = "wandb"
    local_root: str = "./wandb"
    data_root: str | None = None

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


def build_tracker(experiment_config: ExperimentConfig, wandb_run, model):
    run_tracker = None

    if (
        experiment_config.inference.label == "buffer-vi"
        or experiment_config.inference.label == "full-vi"
        or experiment_config.inference.label == "hybrid-vi"
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

        run_tracker = vi.train_bayesian.Tracker(
            record_trigger=make_record_trigger(10),
            metric_samples=experiment_config.test_samples,
            custom_record_fcns=custom_record_fcns,
        )

    if (
        experiment_config.inference.label == "buffer-sgld"
        or experiment_config.inference.label == "particle-mcmc"
    ):
        # down sample block to not do excessive q comp
        down_sample = max(
            experiment_config.inference.config.sample_block_size,
            1000,
        )
        def down_sampled_qs(sample_path):
            q05, q95 = jnp.quantile(
                sample_path[:int(down_sample)], 
                jnp.array([0.05, 0.95])
            )
            mean = jnp.mean(sample_path[:int(down_sample)])
            return mean, q05, q95 
        down_sampled_qs = jax.jit(down_sampled_qs)

        def run_tracker(
            elapsed_time_s,
            num_samples_taken,
            sample_block,
        ):
            update = {
                "elapsed_time_s": elapsed_time_s,
                "num_samples_taken": num_samples_taken,
            }
            model_p = model.parameterization.to_model_parameters(sample_block)
            for f in model.target.parameter_cls.fields():
                mean, q05, q95  = down_sampled_qs(
                    getattr(model_p, f)
                )
                update[f"{f}_q05"] = q05
                update[f"{f}_q95"] = q95
                update[f"{f}_mean"] = mean
            wandb_run.log(update)

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

    data_wandb_run: io.WandbRun | None = None

    model = experiment_config.data_config.posterior

    """
    There are 3 ways to load data.
    1. Local via RealData
    2. Local via WandB artifacts (for offline mode)
    3. Remote via WandB artifacts (for online mode)
    """
    if isinstance(experiment_config.data_config, model_registry.RealDataConfig):
        data_folder = resolved_runtime_config.data_root or resolved_runtime_config.local_root
        prepared_storage = io.LocalPreparedDataStorage(data_folder)
        _, observations, conditions = prepared_storage.get_data(
            experiment_config.data_config,
            experiment_config.data_config,
        )
        x_paths = None
    elif isinstance(experiment_config.data_config, model_registry.SyntheticDataConfig):
        remote_storage: io.DataStorage
        if resolved_runtime_config.wandb_offline:
            remote_storage = io.LocalFilesystemDataStorage(resolved_runtime_config.local_root)
        else:
            data_wandb_run = cast(io.WandbRun, wandb.init(project=experiment_name))
            remote_storage = io.WandbArtifactDataStorage(data_wandb_run)

        x_paths, observations, conditions = remote_storage.get_data(
            experiment_config.data_config
        )
    else:
        raise Exception(f"Unsupported data config type {experiment_config.data_config}")

    
    # Allow special init for NUTS for purpose of generating reference posteriors
    if (
        experiment_config.init_from_generative 
        and experiment_config.inference.label == "NUTS"
        and isinstance(experiment_config.data_config, model_registry.SyntheticDataConfig)
    ):
        generative_params = experiment_config.data_config.generative_parameters
        nuts_config_with_init = replace(
            experiment_config.inference.config,
            initial_latents=x_paths,
            initial_params=model.parameterization.from_model_parameters(generative_params),
        )
        experiment_config = replace(
            experiment_config,
            inference=replace(
                experiment_config.inference,
                config=nuts_config_with_init,
            )
        )

    # Allow special init for PMMH for purpose of generating reference posteriors
    if (
        experiment_config.init_from_generative 
        and experiment_config.inference.label == "particle-mcmc"
        and isinstance(experiment_config.data_config, model_registry.SyntheticDataConfig)
    ):
        generative_params = experiment_config.data_config.generative_parameters
        config_with_init = replace(
            experiment_config.inference.config,
            initial_params=model.parameterization.from_model_parameters(generative_params),
        )
        experiment_config = replace(
            experiment_config,
            inference=replace(
                experiment_config.inference,
                config=config_with_init,
            )
        )

    condition_paths = seqjtyping.NoCondition() if conditions is None else conditions

    dataset = inference_interface.ObservationDataset(
        observations=cast(seqjtyping.Observation, observations),
        conditions=cast(seqjtyping.Condition, condition_paths),
    )

    if data_wandb_run is not None:
        data_wandb_run.finish()

    config_dict = asdict(experiment_config)
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
        key=jrandom.key(experiment_config.fit_seed),
        dataset=dataset,
        test_samples=experiment_config.test_samples,
        config=experiment_config.inference.config,
        tracker=build_tracker(experiment_config, wandb_run, model),
    )

    # end here, to tidy up any stale wandb processes
    # without explicit end+restart I have observed bugs for long
    # running procedures
    wandb_run.finish() 

    process_wandb_run = wandb.init(
        project=experiment_name,
        id=wandb_run.id,
        resume="must",
        settings=wandb.Settings(
            silent=True,
        )
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

def build_buffer_vi_grid_samples(
    experiment: str,
    buffer_vi_run_id: str, 
    sampling_interval_s: int,
):
    print(f"Building grid samples for {buffer_vi_run_id}")
    
    results_run = wandb.init(
        id=buffer_vi_run_id,
        project=experiment,
        settings=wandb.Settings(silent=True),
        resume="must",
    )
    assert results_run.config["inference"]["label"] == "buffer-vi"

    if "generative_parameter_label" in results_run.config['data_config']:
        data_config = model_registry.SyntheticDataConfig(**results_run.config['data_config'])
    else:
        data_config = model_registry.RealDataConfig(**results_run.config['data_config'])

    target_posterior = data_config.posterior

    loaded_samples = io.load_packable_artifact_all(
        results_run, 
        f"{results_run.name}_checkpoint_samples", 
        target_posterior.parameterization.inference_parameter_cls
    )
    
    sample_times = np.array([metadata["elapsed_time_s"] for _, metadata in loaded_samples])
    sample_times_sort_ix = np.argsort(sample_times)

    sample_times = sample_times[sample_times_sort_ix]
    loaded_samples = [loaded_samples[ix] for ix in sample_times_sort_ix]

    grid = np.arange(sampling_interval_s, max(sample_times), sampling_interval_s)
    grid_ix = np.argmin(np.abs(sample_times - grid.reshape(-1, 1)), axis=1)

    grid_samples = [
        (
            f"samples_{grid_point_ix}",
            loaded_samples[ix][0],
            {"elapsed_time_s": float(sample_times[ix])},
        )
        for grid_point_ix, ix in enumerate(grid_ix)
    ]

    return grid_samples