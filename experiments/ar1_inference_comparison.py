import matplotlib.pyplot as plt
import jax
from dataclasses import dataclass, asdict, field
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
import arviz as az
import equinox as eqx
import wandb

from seqjax.model import simulate
from seqjax.model.ar import AR1Target, ARParameters, HalfCauchyStds
from seqjax.inference.particlefilter import (
    BootstrapParticleFilter,
    current_particle_mean,
    current_particle_quantiles,
    run_filter,
    log_marginal,
)

from seqjax.inference import pmcmc, mcmc, particlefilter, sgld, registry

# from seqjax.inference.buffered import BufferedSGLDConfig, run_buffered_sgld
# from seqjax.inference.sgld import SGLDConfig
from seqjax.model import ar
from seqjax.model.typing import HyperParameters, Condition
from seqjax.model.base import BayesianSequentialModel
from seqjax.inference import mcmc, InferenceMethod


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


def analytic_log_marginal(target_posterior, params, y_path):
    def log_marginal(params):
        R = target_posterior.target_parameter(params).observation_std ** 2
        Q = target_posterior.target_parameter(params).transition_std ** 2

        # initial predictive for x₁
        x_pred = 0.0

        P_pred = true_params.transition_std**2

        def scan_step(carry, y_t):
            x_pred, P_pred, ll = carry

            # 1) marginal for y_t
            S = P_pred + R
            ll = ll - 0.5 * (jnp.log(2 * jnp.pi * S) + (y_t - x_pred) ** 2 / S)

            # 2) Kalman update
            K = P_pred / S
            x_filt = x_pred + K * (y_t - x_pred)
            P_filt = (1 - K) * P_pred

            # 3) predict to next
            x_pred_next = target_posterior.target_parameter(params).ar * x_filt
            P_pred_next = target_posterior.target_parameter(params).ar ** 2 * P_filt + Q

            return (x_pred_next, P_pred_next, ll), None

        (x_final, P_final, total_ll), _ = jax.lax.scan(
            scan_step, init=(x_pred, P_pred, 0.0), xs=y_path.y
        )
        return total_ll + target_posterior.parameter_prior.log_prob(params, None)

    return log_marginal(params)


parameter_settings = {
    "base": ar.ARParameters(
        ar=jnp.array(0.8),
        observation_std=jnp.array(0.1),
        transition_std=jnp.array(0.5),
    )
}


@dataclass
class ARExperimentConfig:
    parameter_setting: str
    sequence_length: int
    data_seed: int

    test_samples: int
    fit_seed: int
    inference: registry.InferenceConfig


def run_experiment(experiment_name: str, experiment_config: ARExperimentConfig):
    # track run data
    wandb_run = wandb.init(
        project=experiment_name,
        config=asdict(experiment_config),
    )

    # define target model
    target_params = parameter_settings[experiment_config.parameter_setting]
    model = ar.AR1Bayesian(target_params)

    # generate data
    data_key = jrandom.PRNGKey(experiment_config.data_seed)
    x_path, y_path, _, _ = simulate.simulate(
        data_key,
        model.target,
        None,
        target_params,
        sequence_length=experiment_config.sequence_length,
    )

    # inference init
    inference = registry.build_inference(experiment_config.inference, model)

    elapsed_time_s, _, param_samples, extra_data = inference(
        model,
        hyperparameters=None,
        key=jrandom.key(experiment_config.fit_seed),
        observation_path=y_path,
        condition_path=None,
        test_samples=experiment_config.test_samples,
    )

    artifact = wandb.Artifact(
        name=f"{experiment_config.inference.method}_samples", type="dataset"
    )
    eqx.tree_serialise_leaves("some_filename.eqx", model)

    artifact.add_file(
        local_path=f"./{experiment_config.inference.method}_samples.dat",
        name="training_dataset",
    )
    artifact.save()

    return param_samples

    # ar_sample_sets[label] = param_samples.ar
    # sample_sets[label] = param_samples
    # inference_time[label] = time_s


inference_methods = {
    "NUTS": registry.NUTSInference(
        "NUTS", mcmc.NUTSConfig(step_size=1e-3, num_warmup=100, num_chains=2)
    )
}


if __name__ == "__main__":
    data_repeats = 2
    experiment_name = "seq_rough"
    for data_seed in range(data_repeats):
        for label, inference_config in inference_methods.items():
            experiment_config = ARExperimentConfig(
                parameter_setting="base",
                sequence_length=100,
                test_samples=100,
                data_seed=data_seed,
                fit_seed=1000,
                inference=inference_config,
            )
            run_experiment(experiment_name, experiment_config)

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

    # ar_sample_sets = {}
    # sample_sets = {}
    # inference_time = {}
    # for label, procedure in inference_procedures.items():
    #     print(f"Running: {label}")
    #     time_s, _, param_samples = procedure(
    #         model,
    #         hyperparameters=None,
    #         key=jrandom.key(100),
    #         observation_path=y_path,
    #         condition_path=None,
    #     )
    #     ar_sample_sets[label] = param_samples.ar
    #     sample_sets[label] = param_samples
    #     inference_time[label] = time_s

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

    # plt.figure(figsize=(8, 8))
    # plt.title("MSE vs Inference time")
    # for label in ar_sample_sets:
    #     ar_samples = ar_sample_sets[label]
    #     time_s = inference_time[label]
    #     plt.plot(
    #         time_s,
    #         jnp.cumsum(jnp.square(ar_samples - true_params.ar))
    #         / jnp.arange(1, 1 + len(ar_samples)),
    #         label=label,
    #     )
    # plt.legend()
    # plt.xlabel("Inference Time (s)")
    # plt.ylabel("Parameter MSE")
    # plt.grid()
    # plt.savefig("parameter_mse_path_comparison.png")
    # plt.show()

    # for mcmc_sampler in mcmc_samplers:
    #     if mcmc_sampler in ar_sample_sets:
    #         plt.figure(figsize=(8, 3))
    #         plt.title(f"{mcmc_sampler} path")

    #         nuts_ar1_samples = sample_sets[mcmc_sampler]
    #         time_s = inference_time[mcmc_sampler]

    #         plt.plot(
    #             time_s,
    #             nuts_ar1_samples.ar,
    #             label=label,
    #         )

    #         plt.legend()
    #         plt.xlabel("Inference Time (s)")
    #         plt.ylabel("AR1")
    #         plt.grid()
    #         plt.savefig(f"{mcmc_sampler}_sample_path.png")
    #         plt.show()

    #         plt.figure(figsize=(8, 3))
    #         az.plot_autocorr(asdict(nuts_ar1_samples))
    #         plt.grid()
    #         plt.savefig(f"{mcmc_sampler}_acf.png")
    #         plt.show()
