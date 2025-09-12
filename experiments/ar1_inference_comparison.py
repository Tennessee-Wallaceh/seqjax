import matplotlib.pyplot as plt
import jax
from dataclasses import dataclass, asdict
import jax.numpy as jnp
import jax.random as jrandom
import arviz as az
import wandb

from seqjax import util
from seqjax.model import registry as model_registry

from seqjax.inference import vi
from seqjax.inference import registry as inference_registry
from seqjax import io

# from seqjax.inference.buffered import BufferedSGLDConfig, run_buffered_sgld
# from seqjax.inference.sgld import SGLDConfig
from seqjax.model import ar


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


@dataclass
class ARExperimentConfig:
    data_config: model_registry.ARDataConfig
    test_samples: int
    fit_seed: int
    inference: inference_registry.InferenceConfig


def process_mcmc_results(
    run,
    experiment_config,
    elapsed_time_s,
    latent_samples,
    param_samples,
    extra_data,
    x_path,
    y_path,
):
    experiment_shorthand = f"{experiment_config.inference.name} {experiment_config.data_config.dataset_name}"
    io.save_packable_artifact(
        run,
        f"{run.name}_samples",
        "run_output",
        [("final_samples", param_samples, {})],
    )

    # plot hist of
    fig = plt.figure(figsize=(8, 3))
    generative_params = experiment_config.data_config.generative_parameters
    plt.hist(param_samples.ar, bins=50, density=True, alpha=0.5, label=label)
    plt.axvline(generative_params.ar, color="black", linestyle="--", label="true")
    plt.xlabel("ar parameter")
    plt.ylabel("density")
    plt.legend()
    plt.grid()
    plt.title(f"{experiment_shorthand} samples")
    plt.tight_layout()

    run.log({"final_sample": wandb.Image(fig)})
    plt.close(fig)

    # sample plot
    fig = plt.figure(figsize=(8, 3))
    plt.title(f"{experiment_shorthand} sample path")

    plt.plot(
        elapsed_time_s,
        param_samples.ar,
        label=label,
    )

    plt.legend()
    plt.xlabel("Inference Time (s)")
    plt.ylabel("AR")
    plt.grid()

    run.log({"sample_path": wandb.Image(fig)})
    plt.close(fig)

    # autocorr plot
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
    az.plot_autocorr(asdict(param_samples), ax=ax)
    plt.title(f"{experiment_shorthand} autocorrelation")
    plt.tight_layout()
    run.log({"autocorrelation": wandb.Image(fig)})
    plt.close(fig)

    # generative p MSE
    fig = plt.figure(figsize=(8, 8))
    plt.title(f"{experiment_shorthand} MSE to generative vs inference time")
    plt.plot(
        elapsed_time_s,
        jnp.cumsum(jnp.square(param_samples.ar - generative_params.ar))
        / jnp.arange(1, 1 + len(param_samples.ar)),
    )
    plt.xlabel("Inference Time (s)")
    plt.ylabel("Parameter MSE")
    plt.grid()
    run.log({"parameter_mse_plot": wandb.Image(fig)})
    plt.close(fig)
    run.log(
        {
            "generative_parameter_mse": jnp.mean(
                jnp.square(param_samples.ar - generative_params.ar)
            )
        }
    )

    # latent fit
    fig = plt.figure(figsize=(8, 3))
    plt.plot(x_path.x, linestyle="--", c="black", label="true latent")
    for ix in range(5):
        latent_sample = util.index_pytree(latent_samples, (-ix, 0))
        plt.plot(latent_sample.x, c="blue", alpha=0.5)
    plt.grid()
    plt.ylabel("x")
    plt.ylabel("t")
    plt.title(f"{experiment_shorthand} Latent Approximation")
    run.log({"latent_approximation": wandb.Image(fig)})
    plt.close(fig)

    # save down samples
    num_sample_points = 10
    num_samples = param_samples.batch_shape[0]
    block_size = int(num_samples / num_sample_points)

    io.save_packable_artifact(
        run,
        f"{run.name}_checkpoint_samples",
        "checkpoint_samples",
        [
            (
                f"samples_{i}",
                util.slice_pytree(param_samples, 0, block_size * i),
                {"elapsed_time_s": float(elapsed_time_s[i * block_size])},
            )
            for i in range(1, num_sample_points + 1)
        ],
    )


def process_buffer_vi(
    run,
    experiment_config,
    elapsed_time_s,
    latent_samples,
    param_samples,
    extra_data,
    x_path,
    y_path,
):
    run_data, approx_start, run_tracker = extra_data
    experiment_shorthand = f"{experiment_config.inference.name} {experiment_config.data_config.dataset_name}"
    io.save_packable_artifact(
        run,
        f"{run.name}_samples",
        "run_output",
        [("final_samples", param_samples, {})],
    )

    fig = plt.figure(figsize=(8, 3))
    generative_params = experiment_config.data_config.generative_parameters
    plt.hist(param_samples.ar, bins=50, density=True, alpha=0.5, label=label)
    plt.axvline(generative_params.ar, color="black", linestyle="--", label="true")
    plt.xlabel("ar parameter")
    plt.ylabel("density")
    plt.legend()
    plt.grid()
    plt.title(f"{experiment_shorthand} samples")
    plt.tight_layout()
    run.log({"final_sample": wandb.Image(fig)})
    plt.close(fig)

    run.log(
        {
            "generative_parameter_mse": jnp.mean(
                jnp.square(param_samples.ar - generative_params.ar)
            )
        }
    )

    fig = plt.figure(figsize=(8, 3))
    plt.plot(run_data["elapsed_time_s"], run_data["ar_q05"], c="green")
    plt.plot(run_data["elapsed_time_s"], run_data["ar_q95"], c="blue", linestyle="--")
    plt.axhline(generative_params.ar, c="black")
    plt.grid()
    plt.title(f"{experiment_shorthand} quantiles")
    plt.xlabel("Elapsed time (s)")
    run.log({"quantile_plot": wandb.Image(fig)})
    plt.close(fig)

    # approximation plot
    fig = plt.figure(figsize=(8, 3))
    plt.title(f"{experiment_shorthand} latent approximation")
    for start_sample_ix in range(5):
        start_ix = approx_start[start_sample_ix]

        for sample_ix in range(3):
            latent_sample = util.index_pytree(
                latent_samples, (start_sample_ix, sample_ix)
            )
            plt.plot(
                range(start_ix, start_ix + len(latent_sample.x)),
                latent_sample.x,
                c="blue",
                alpha=0.5,
            )
            plt.scatter(start_ix, latent_sample.x[0], marker="x", c="blue")

    plt.ylabel("x")
    plt.ylabel("t")
    plt.plot(x_path.x, c="black", linestyle="--")
    plt.grid()
    run.log({"latent_approximation": wandb.Image(fig)})
    plt.close(fig)

    # save checkpoint samples
    io.save_packable_artifact(
        run,
        f"{run.name}_checkpoint_samples",
        "checkpoint_samples",
        [
            (
                f"samples_{i}",
                samples,
                {"elapsed_time_s": elapsed_time_s},
            )
            for i, (elapsed_time_s, samples) in enumerate(
                run_tracker.checkpoint_samples
            )
        ],
    )

    #
    fig = plt.figure(figsize=(8, 3))
    plt.plot(run_data["elapsed_time_s"], run_data["loss"], c="green")
    plt.grid()
    plt.title(f"{experiment_shorthand} loss")
    plt.xlabel("Elapsed time (s)")
    run.log({"loss_plot": wandb.Image(fig)})
    plt.close(fig)


def process_full_vi(
    run,
    experiment_config,
    elapsed_time_s,
    latent_samples,
    param_samples,
    extra_data,
    x_path,
    y_path,
):
    run_data, run_tracker = extra_data
    experiment_shorthand = f"{experiment_config.inference.name} {experiment_config.data_config.dataset_name}"
    io.save_packable_artifact(
        run,
        f"{run.name}_samples",
        "run_output",
        [("final_samples", param_samples, {})],
    )

    fig = plt.figure(figsize=(8, 3))
    generative_params = experiment_config.data_config.generative_parameters
    plt.hist(param_samples.ar, bins=50, density=True, alpha=0.5, label=label)
    plt.axvline(generative_params.ar, color="black", linestyle="--", label="true")
    plt.xlabel("ar parameter")
    plt.ylabel("density")
    plt.legend()
    plt.grid()
    plt.title(f"{experiment_shorthand} samples")
    plt.tight_layout()
    run.log({"final_sample": wandb.Image(fig)})
    plt.close(fig)

    run.log(
        {
            "generative_parameter_mse": jnp.mean(
                jnp.square(param_samples.ar - generative_params.ar)
            )
        }
    )

    fig = plt.figure(figsize=(8, 3))
    plt.plot(run_data["elapsed_time_s"], run_data["ar_q05"], c="green")
    plt.plot(run_data["elapsed_time_s"], run_data["ar_q95"], c="blue", linestyle="--")
    plt.axhline(generative_params.ar, c="black")
    plt.grid()
    plt.title(f"{experiment_shorthand} quantiles")
    plt.xlabel("Elapsed time (s)")
    run.log({"quantile_plot": wandb.Image(fig)})
    plt.close(fig)

    # approximation plot
    fig = plt.figure(figsize=(8, 3))
    plt.title(f"{experiment_shorthand} latent approximation")
    for sample_ix in range(5):
        latent_sample = util.index_pytree(latent_samples, (0, sample_ix))
        plt.plot(
            latent_sample.x,
            c="blue",
            alpha=0.5,
        )

    plt.ylabel("x")
    plt.ylabel("t")
    plt.plot(x_path.x, c="black", linestyle="--")
    plt.grid()
    run.log({"latent_approximation": wandb.Image(fig)})
    plt.close(fig)

    # save checkpoint samples
    io.save_packable_artifact(
        run,
        f"{run.name}_checkpoint_samples",
        "checkpoint_samples",
        [
            (
                f"samples_{i}",
                samples,
                {"elapsed_time_s": elapsed_time_s},
            )
            for i, (elapsed_time_s, samples) in enumerate(
                run_tracker.checkpoint_samples
            )
        ],
    )

    #
    fig = plt.figure(figsize=(8, 3))
    plt.plot(run_data["elapsed_time_s"], run_data["loss"], c="green")
    plt.grid()
    plt.title(f"{experiment_shorthand} loss")
    plt.xlabel("Elapsed time (s)")
    run.log({"loss_plot": wandb.Image(fig)})
    plt.close(fig)


results_process = {
    "NUTS": process_mcmc_results,
    "buffer-vi": process_buffer_vi,
    "full-vi": process_full_vi,
}


def process_results(
    run,
    experiment_config,
    elapsed_time_s,
    latent_samples,
    param_samples,
    extra_data,
    x_path,
    y_path,
):
    result_processor = results_process.get(experiment_config.inference.method, None)
    if result_processor is not None:
        result_processor(
            run,
            experiment_config,
            elapsed_time_s,
            latent_samples,
            param_samples,
            extra_data,
            x_path,
            y_path,
        )


def run_experiment(experiment_name: str, experiment_config: ARExperimentConfig):
    # track run data
    wandb_run = wandb.init(
        project=experiment_name,
        config={
            **asdict(experiment_config),
            "inference_name": experiment_config.inference.name,
        },  # force inference name into config
    )

    # define target model
    target_params = experiment_config.data_config.generative_parameters
    model = ar.AR1Bayesian(target_params)

    # get target data
    x_path, y_path = io.get_remote_data(wandb_run, experiment_config.data_config)

    # inference init
    inference = inference_registry.build_inference(experiment_config.inference, model)

    elapsed_time_s, latent_samples, param_samples, extra_data = inference(
        model,
        hyperparameters=None,
        key=jrandom.key(experiment_config.fit_seed),
        observation_path=y_path,
        condition_path=None,
        test_samples=experiment_config.test_samples,
    )

    process_results(
        wandb_run,
        experiment_config,
        elapsed_time_s,
        latent_samples,
        param_samples,
        extra_data,
        x_path,
        y_path,
    )

    wandb_run.finish()

    return (elapsed_time_s, latent_samples, param_samples, extra_data, x_path, y_path)


"""
Select inference methods to run
"""
inference_methods = {}

# inference_methods["NUTS"] = inference_registry.NUTSInference(
#     "NUTS", mcmc.NUTSConfig(step_size=1e-3, num_warmup=1000, num_chains=1)
# )

for buffer in [2, 5]:
    for batch in [10]:
        for cv in [True]:
            # for ar_transform in ["interval_spline"]:
            for ar_transform in ["sigmoid"]:
                buffviconf = inference_registry.BufferVI(
                    "buffer-vi",
                    vi.BufferedVIConfig(
                        learning_rate=2e-3,
                        opt_steps=20000,
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
#         learning_rate=2e-3,
#         opt_steps=20000,
#         parameter_field_bijections={"ar": "sigmoid"},
#     ),
# )
# inference_methods[full_vi_config.name] = full_vi_config

if __name__ == "__main__":
    data_repeats = 1
    experiment_name = "ar1-experimental"
    for data_seed in range(data_repeats):
        for label, inference_config in inference_methods.items():
            experiment_config = ARExperimentConfig(
                data_config=model_registry.ARDataConfig(
                    generative_parameter_label="base",
                    sequence_length=1000,
                    seed=data_seed,
                ),
                test_samples=10000,
                fit_seed=1000,
                inference=inference_config,
            )
            output = run_experiment(experiment_name, experiment_config)

            # try:
            #     output = run_experiment(experiment_name, experiment_config)

            # except Exception as e:
            #     print(e)
            #     pass
    # define inference procedures
    # inference_procedures = {}
    # mcmc_samplers = ["NUTS", "PMMH", "full_SGLD"]

    # inference_procedures["NUTS"] =

    # inference_procedures["PMMH"] = partial(
    #     pmcmc.run_particle_mcmc,
    #     config=pmcmc.ParticleMCMCConfig(
    #         mcmc=mcmc.RandomWalkConfig(5e-2, samples),
    #         particle_filter=particlefilter.BootstrapParticleFilter(
    #             model.target,
    #             num_particles=10000,
    #             ess_threshold=1.5,
    #             target_parameters=model.target_parameter,
    #         ),
    #         initial_parameter_guesses=20,
    #     ),
    # )

    # inference_procedures["full_SGLD"] = partial(
    #     sgld.run_full_sgld_mcmc,
    #     config=sgld.SGLDConfig(
    #         particle_filter=particlefilter.BootstrapParticleFilter(
    #             model.target,
    #             num_particles=10000,
    #             ess_threshold=-0.5,
    #             target_parameters=model.target_parameter,
    #         ),
    #         step_size=5e-3,
    #         num_samples=samples,
    #         initial_parameter_guesses=20,
    #     ),
    # )

    # min_ar = 1
    # max_ar = -1
    # print(f"TRUE: {true_params.ar:.2f}")
    # for label, ar_set in ar_sample_sets.items():
    #     q05, q95 = jnp.quantile(ar_set, jnp.array([0.05, 0.95]))
    #     print(f"{label}: {jnp.mean(ar_set):.2f} ({q05:.2f}, {q95:.2f})")
    #     min_ar = min(min_ar, jnp.min(ar_set))
    #     max_ar = max(max_ar, jnp.max(ar_set))

    # plt.figure(figsize=(6, 3))
    # bins = jnp.linspace(min_ar, max_ar, 100)

    # candidate_ar = jnp.linspace(-1, 1, 1000)

    # inspect_p = jax.vmap(
    #     lambda x: ar.AROnlyParameters(
    #         ar=x,
    #     )
    # )(candidate_ar)

    # log_marginal = jax.vmap(analytic_log_marginal, in_axes=[None, 0, None])(
    #     model, inspect_p, y_path
    # )

    # plt.plot(
    #     candidate_ar,
    #     jnp.exp(log_marginal)
    #     / jax.scipy.integrate.trapezoid(jnp.exp(log_marginal), candidate_ar),
    # )

    # for label, ar_set in ar_sample_sets.items():
    #     plt.hist(ar_set, bins=bins, density=True, alpha=0.5, label=label)

    # plt.xlim([min_ar, max_ar])
    # plt.axvline(true_params.ar, color="black", linestyle="--", label="true")
    # plt.xlabel("ar parameter")
    # plt.ylabel("density")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("approximate_posterior_comparison.png")
    # plt.show()
