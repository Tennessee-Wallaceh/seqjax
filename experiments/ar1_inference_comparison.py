import matplotlib.pyplot as plt
import jax
from dataclasses import asdict
import jax.numpy as jnp
import arviz as az
import wandb
import polars as pl


from experiments.core import ExperimentConfig, ResultProcessor, run_experiment
from seqjax import util
from seqjax.model import registry as model_registry

from seqjax.inference import vi
from seqjax.inference import registry as inference_registry
from seqjax import io

# from seqjax.inference.buffered import BufferedSGLDConfig, run_buffered_sgld
# from seqjax.inference.sgld import SGLDConfig


def cumulative_quantiles_masked(samples, quantiles):
    # High memory quantile computation
    n = samples.shape[0]

    # Pad samples to (n, n) matrix of prefixes
    full_samples = jnp.broadcast_to(samples, (n, n))
    mask = jnp.tril(jnp.ones((n, n)))  # Lower-triangular mask

    def compute_row(row, mask_row):
        valid = jnp.where(mask_row == 1, row, jnp.nan)
        return jnp.nanquantile(valid, quantiles, axis=-1)

    return jax.vmap(compute_row)(full_samples, mask)

class ARResultProcessor:
    def process(
        self,
        run,
        experiment_config,
        param_samples,
        extra_data,
        x_path,
        observation_path,
        condition=None,
    ) -> None:
        experiment_shorthand = (
            f"{experiment_config.inference.name} "
            f"{experiment_config.data_config.dataset_name}"
        )
        label = experiment_config.inference.name
        generative_params = experiment_config.data_config.generative_parameters

        io.save_packable_artifact(
            run,
            f"{run.name}_samples",
            "run_output",
            [("final_samples", param_samples, {})],
        )

        method = experiment_config.inference.method
        if extra_data is None:
            raise ValueError(
                f"Extra data required to process results for {method} inference."
            )

        if method == "NUTS":
            try:
                elapsed_time_s, latent_samples = extra_data
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Expected extra data to be (elapsed_time_s, latent_samples) "
                    "for NUTS results."
                ) from exc
            self._process_mcmc(
                run,
                experiment_shorthand,
                label,
                param_samples,
                generative_params,
                elapsed_time_s,
                latent_samples,
                x_path,
            )
        elif method == "buffer-vi":
            try:
                approx_start, latent_samples, run_tracker = extra_data
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Expected extra data to be "
                    "(approx_start, latent_samples, run_tracker) "
                    "for buffered VI results."
                ) from exc
            self._process_buffer_vi(
                run,
                experiment_shorthand,
                label,
                param_samples,
                generative_params,
                approx_start,
                latent_samples,
                run_tracker,
                x_path,
            )
        elif method == "full-vi":
            try:
                run_tracker, latent_samples = extra_data
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Expected extra data to be (run_tracker, latent_samples) "
                    "for full VI results."
                ) from exc
            self._process_full_vi(
                run,
                experiment_shorthand,
                label,
                param_samples,
                generative_params,
                run_tracker,
                latent_samples,
                x_path,
            )
        else:
            raise ValueError(
                f"Unsupported inference method '{method}' for ARResultProcessor."
            )

    def _log_parameter_histogram(
        self,
        run,
        experiment_shorthand,
        label,
        param_samples,
        generative_params,
    ) -> None:
        fig = plt.figure(figsize=(8, 3))
        samples = jnp.asarray(param_samples.ar)
        plt.hist(samples, bins=50, density=True, alpha=0.5, label=label)
        plt.axvline(
            float(jnp.asarray(generative_params.ar)),
            color="black",
            linestyle="--",
            label="true",
        )
        plt.xlabel("ar parameter")
        plt.ylabel("density")
        plt.legend()
        plt.grid()
        plt.title(f"{experiment_shorthand} samples")
        plt.tight_layout()
        run.log({"final_sample": wandb.Image(fig)})
        plt.close(fig)

    def _log_parameter_mse_scalar(
        self,
        run,
        param_samples,
        generative_params,
    ) -> None:
        mse = float(
            jnp.mean(
                jnp.square(
                    jnp.asarray(param_samples.ar) - jnp.asarray(generative_params.ar)
                )
            )
        )
        run.log({"generative_parameter_mse": mse})

    def _process_mcmc(
        self,
        run,
        experiment_shorthand,
        label,
        param_samples,
        generative_params,
        elapsed_time_s,
        latent_samples,
        x_path,
    ) -> None:
        ar_values = jnp.asarray(param_samples.ar)
        times = jnp.asarray(elapsed_time_s)

        self._log_parameter_histogram(
            run, experiment_shorthand, label, param_samples, generative_params
        )
        self._log_parameter_mse_scalar(run, param_samples, generative_params)

        if ar_values.size and times.size:
            fig = plt.figure(figsize=(8, 3))
            plt.title(f"{experiment_shorthand} sample path")
            plt.plot(times, ar_values, label=label)
            plt.legend()
            plt.xlabel("Inference Time (s)")
            plt.ylabel("AR")
            plt.grid()
            run.log({"sample_path": wandb.Image(fig)})
            plt.close(fig)

        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        az.plot_autocorr(asdict(param_samples), ax=ax)
        plt.title(f"{experiment_shorthand} autocorrelation")
        run.log({"autocorrelation": wandb.Image(fig)})
        plt.close(fig)

        if ar_values.size and times.size:
            mse_curve = jnp.cumsum(
                jnp.square(ar_values - jnp.asarray(generative_params.ar))
            ) / jnp.arange(1, ar_values.shape[0] + 1)
            fig = plt.figure(figsize=(8, 8))
            plt.title(f"{experiment_shorthand} MSE to generative vs inference time")
            plt.plot(times, mse_curve)
            plt.xlabel("Inference Time (s)")
            plt.ylabel("Parameter MSE")
            plt.grid()
            run.log({"parameter_mse_plot": wandb.Image(fig)})
            plt.close(fig)

        fig = plt.figure(figsize=(8, 3))
        plt.plot(jnp.asarray(x_path.x), linestyle="--", c="black", label="true latent")
        for ix in range(5):
            try:
                latent_sample = util.index_pytree(latent_samples, (-ix, 0))
            except Exception:
                break
            plt.plot(jnp.asarray(latent_sample.x), c="blue", alpha=0.5)
        plt.grid()
        plt.ylabel("x")
        plt.ylabel("t")
        plt.title(f"{experiment_shorthand} Latent Approximation")
        run.log({"latent_approximation": wandb.Image(fig)})
        plt.close(fig)

        checkpoint_entries = self._create_mcmc_checkpoint_entries(param_samples, times)
        if checkpoint_entries:
            io.save_packable_artifact(
                run,
                f"{run.name}_checkpoint_samples",
                "checkpoint_samples",
                checkpoint_entries,
            )

    def _process_buffer_vi(
        self,
        run,
        experiment_shorthand,
        label,
        param_samples,
        generative_params,
        approx_start,
        latent_samples,
        run_tracker,
        x_path,
    ) -> None:
        self._log_parameter_histogram(
            run, experiment_shorthand, label, param_samples, generative_params
        )
        self._log_parameter_mse_scalar(run, param_samples, generative_params)

        run_data = self._get_run_data(run_tracker)
        columns = set(run_data.columns)

        if run_data.height > 0 and {"elapsed_time_s", "ar_q05", "ar_q95"}.issubset(
            columns
        ):
            fig = plt.figure(figsize=(8, 3))
            plt.plot(
                run_data["elapsed_time_s"].to_numpy(),
                run_data["ar_q05"].to_numpy(),
                c="green",
            )
            plt.plot(
                run_data["elapsed_time_s"].to_numpy(),
                run_data["ar_q95"].to_numpy(),
                c="blue",
                linestyle="--",
            )
            plt.axhline(float(jnp.asarray(generative_params.ar)), c="black")
            plt.grid()
            plt.title(f"{experiment_shorthand} quantiles")
            plt.xlabel("Elapsed time (s)")
            run.log({"quantile_plot": wandb.Image(fig)})
            plt.close(fig)

        fig = plt.figure(figsize=(8, 3))
        plt.title(f"{experiment_shorthand} latent approximation")
        approx_indices = jnp.asarray(approx_start).ravel().tolist()
        for start_sample_ix, start_ix in enumerate(approx_indices[:5]):
            for sample_ix in range(3):
                try:
                    latent_sample = util.index_pytree(
                        latent_samples, (start_sample_ix, sample_ix)
                    )
                except Exception:
                    break
                time_axis = range(int(start_ix), int(start_ix) + len(latent_sample.x))
                plt.plot(time_axis, jnp.asarray(latent_sample.x), c="blue", alpha=0.5)
                if len(latent_sample.x):
                    plt.scatter(
                        int(start_ix),
                        float(latent_sample.x[0]),
                        marker="x",
                        c="blue",
                    )
        plt.ylabel("x")
        plt.ylabel("t")
        plt.plot(jnp.asarray(x_path.x), c="black", linestyle="--")
        plt.grid()
        run.log({"latent_approximation": wandb.Image(fig)})
        plt.close(fig)

        checkpoint_samples = getattr(run_tracker, "checkpoint_samples", [])
        if checkpoint_samples:
            io.save_packable_artifact(
                run,
                f"{run.name}_checkpoint_samples",
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

        if run_data.height > 0 and {"elapsed_time_s", "loss"}.issubset(columns):
            fig = plt.figure(figsize=(8, 3))
            plt.plot(
                run_data["elapsed_time_s"].to_numpy(),
                run_data["loss"].to_numpy(),
                c="green",
            )
            plt.grid()
            plt.title(f"{experiment_shorthand} loss")
            plt.xlabel("Elapsed time (s)")
            run.log({"loss_plot": wandb.Image(fig)})
            plt.close(fig)

    def _process_full_vi(
        self,
        run,
        experiment_shorthand,
        label,
        param_samples,
        generative_params,
        run_tracker,
        latent_samples,
        x_path,
    ) -> None:
        self._log_parameter_histogram(
            run, experiment_shorthand, label, param_samples, generative_params
        )
        self._log_parameter_mse_scalar(run, param_samples, generative_params)

        run_data = self._get_run_data(run_tracker)
        columns = set(run_data.columns)

        if run_data.height > 0 and {"elapsed_time_s", "ar_q05", "ar_q95"}.issubset(
            columns
        ):
            fig = plt.figure(figsize=(8, 3))
            plt.plot(
                run_data["elapsed_time_s"].to_numpy(),
                run_data["ar_q05"].to_numpy(),
                c="green",
            )
            plt.plot(
                run_data["elapsed_time_s"].to_numpy(),
                run_data["ar_q95"].to_numpy(),
                c="blue",
                linestyle="--",
            )
            plt.axhline(float(jnp.asarray(generative_params.ar)), c="black")
            plt.grid()
            plt.title(f"{experiment_shorthand} quantiles")
            plt.xlabel("Elapsed time (s)")
            run.log({"quantile_plot": wandb.Image(fig)})
            plt.close(fig)

        fig = plt.figure(figsize=(8, 3))
        plt.title(f"{experiment_shorthand} latent approximation")
        for sample_ix in range(5):
            try:
                latent_sample = util.index_pytree(latent_samples, (0, sample_ix))
            except Exception:
                break
            plt.plot(jnp.asarray(latent_sample.x), c="blue", alpha=0.5)
        plt.ylabel("x")
        plt.ylabel("t")
        plt.plot(jnp.asarray(x_path.x), c="black", linestyle="--")
        plt.grid()
        run.log({"latent_approximation": wandb.Image(fig)})
        plt.close(fig)

        checkpoint_samples = getattr(run_tracker, "checkpoint_samples", [])
        if checkpoint_samples:
            io.save_packable_artifact(
                run,
                f"{run.name}_checkpoint_samples",
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

        if run_data.height > 0 and {"elapsed_time_s", "loss"}.issubset(columns):
            fig = plt.figure(figsize=(8, 3))
            plt.plot(
                run_data["elapsed_time_s"].to_numpy(),
                run_data["loss"].to_numpy(),
                c="green",
            )
            plt.grid()
            plt.title(f"{experiment_shorthand} loss")
            plt.xlabel("Elapsed time (s)")
            run.log({"loss_plot": wandb.Image(fig)})
            plt.close(fig)

    def _create_mcmc_checkpoint_entries(self, param_samples, times):
        ar_values = jnp.asarray(param_samples.ar)
        if ar_values.ndim == 0:
            return []
        num_samples = int(ar_values.shape[0])
        if num_samples == 0:
            return []
        num_sample_points = 10
        block_size = max(1, num_samples // num_sample_points)
        times = jnp.asarray(times)
        if times.ndim == 0:
            time_count = 1
        else:
            time_count = int(times.shape[0])
        entries = []
        for i in range(1, num_sample_points + 1):
            limit = min(block_size * i, num_samples)
            if limit <= 0:
                continue
            time_index = min(limit - 1, time_count - 1) if time_count else 0
            elapsed = float(times[time_index]) if time_count else 0.0
            entries.append(
                (
                    f"samples_{i}",
                    util.slice_pytree(param_samples, 0, limit),
                    {"elapsed_time_s": elapsed},
                )
            )
        return entries

    def _get_run_data(self, run_tracker):
        if run_tracker is None:
            return pl.DataFrame()
        rows = getattr(run_tracker, "update_rows", None)
        if rows is None:
            return pl.DataFrame()
        return pl.DataFrame(rows)
"""
Select inference methods to run
"""
inference_methods = {}

# inference_methods["NUTS"] = inference_registry.NUTSInference(
#     "NUTS", mcmc.NUTSConfig(step_size=1e-3, num_warmup=1000, num_chains=1)
# )

for buffer in [10, 50]:
    for batch in [10]:
        for lr in [1e-2]:
            for cv in [False]:
                # for ar_transform in ["interval_spline"]:
                for ar_transform in ["sigmoid"]:
                    buffviconf = inference_registry.BufferVI(
                        "buffer-vi",
                        vi.BufferedVIConfig(
                            optimization=vi.run.CosineOpt(
                                peak_lr=lr,
                                end_lr=1e-5,
                                warmup_steps=1000,
                                decay_steps=19_000,
                                total_steps=50_000,
                            ),
                            buffer_length=buffer,
                            batch_length=batch,
                            parameter_field_bijections={
                                "ar": ar_transform,
                            },
                            control_variate=cv,
                        ),
                    )
                    inference_methods[buffviconf.name] = buffviconf


# full_vi_config = inference_registry.FullVI(
#     "full-vi",
#     vi.FullVIConfig(
#         learning_rate=3e-3,
#         opt_steps=20000,
#         parameter_field_bijections={"ar": "sigmoid"},
#     ),
# )
# inference_methods[full_vi_config.name] = full_vi_config

if __name__ == "__main__":
    data_repeats = 1
    experiment_name = "ar1-experimental"
    result_processor = ARResultProcessor()
    for fit_seed in [1000]:
        # for fit_seed in [1000]:
        for data_seed in range(data_repeats):
            for _label, inference_config in inference_methods.items():
                experiment_config = ExperimentConfig(
                    data_config=model_registry.ARDataConfig(
                        generative_parameter_label="base",
                        sequence_length=1000,
                        seed=data_seed,
                    ),
                    test_samples=10000,
                    fit_seed=fit_seed,
                    inference=inference_config,
                )
                output = run_experiment(
                    experiment_name,
                    experiment_config,
                    result_processor=result_processor,
                )
