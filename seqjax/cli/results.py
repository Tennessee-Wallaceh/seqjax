from seqjax import io, util
from seqjax.experiment import ExperimentConfig

class ResultProcessor:
    def process(
        self,
        wandb_run,
        experiment_config: ExperimentConfig,
        param_samples,
        extra_data,
        x_path,
        observation_path,
        condition,
    ) -> None:
        if experiment_config.inference.label == "NUTS":
            _, latent_samples, full_param_samples = extra_data

            param_sample_chains = [ #type: ignore
                (
                    f"full_param_samples_c{chain}",
                    util.index_pytree_in_dim(full_param_samples, dim=1, index=chain),
                    {},
                )
                for chain in range(full_param_samples.batch_shape[1])
            ]

            io.save_packable_artifact(
                wandb_run,
                f"{wandb_run.name}-samples",
                "run_output",
                [
                    ("final_samples", param_samples, {}),
                ]
                + param_sample_chains,
            )

            return
        elif experiment_config.inference.label == "buffer-vi":
            approx_start, x_q, run_tracker, fitted_approximation, opt_state = extra_data
        elif experiment_config.inference.label == "full-vi":
            run_tracker, x_q, fitted_approximation, opt_state = extra_data

        elif (
            experiment_config.inference.label == "buffer-sgld" 
            or experiment_config.inference.label == "particle-mcmc" 
        ):
            block_times_s = extra_data

            io.save_packable_artifact(
                wandb_run,
                f"{wandb_run.name}-samples",
                "run_output",
                [
                    ("all_samples", param_samples, {}),
                ]
            )

            io.save_python_artifact(
                wandb_run,
                f"{wandb_run.name}-timings",
                "run_output",
                [
                    ("block_times_s", block_times_s),
                ]
            )

            return
        else:
            raise ValueError(f"Unknown inference method. {experiment_config}")

        # save final model
        io.save_model_artifact(
            wandb_run,
            f"{wandb_run.name}-fitted-approximation",
            fitted_approximation,
        )

        io.save_model_artifact(
            wandb_run,
            f"{wandb_run.name}-optimization-state",
            opt_state,
        )

        checkpoint_samples = getattr(run_tracker, "checkpoint_samples", [])
        if checkpoint_samples:
            io.save_packable_artifact(
                wandb_run,
                f"{wandb_run.name}_checkpoint_samples",
                "checkpoint_samples",
                [
                    (
                        f"samples_{i}",
                        samples,
                        {"elapsed_time_s": float(elapsed_time_s)},
                    )
                    for i, (elapsed_time_s, samples) in enumerate(checkpoint_samples)
                ],
            )
