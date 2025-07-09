import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial

from seqjax.model import simulate
from seqjax.model.ar import AR1Target, ARParameters, HalfCauchyStds
from seqjax.inference.particlefilter import (
    BootstrapParticleFilter,
    current_particle_mean,
    current_particle_quantiles,
    run_filter,
    log_marginal,
)

# from seqjax.inference.buffered import BufferedSGLDConfig, run_buffered_sgld
# from seqjax.inference.sgld import SGLDConfig
from seqjax.model import ar
from seqjax.model.typing import HyperParameters, Condition
from seqjax.model.base import BayesianSequentialModel
from seqjax.inference import mcmc


if __name__ == "__main__":
    # define model
    true_params = ar.ARParameters(
        ar=jnp.array(0.8),
        observation_std=jnp.array(0.1),
        transition_std=jnp.array(0.5),
    )

    model = ar.AR1Bayesian(true_params)

    # generate data
    sequence_length = 50

    key = jrandom.PRNGKey(0)
    x_path, y_path, _, _ = simulate.simulate(
        key, model.target, None, true_params, sequence_length=sequence_length
    )

    # define inference procedures
    inference_procedures = {}

    inference_procedures["NUTS"] = partial(
        mcmc.run_bayesian_nuts,
        config=mcmc.NUTSConfig(
            step_size=1e-5, num_warmup=5000, num_samples=2000, num_chains=2
        ),
    )

    ar_sample_sets = {}
    for label, procedure in inference_procedures.items():
        print(f"Running: {label}")
        latent_samples, param_samples = procedure(
            model,
            hyperparameters=None,
            key=jrandom.key(100),
            observation_path=y_path,
            condition_path=None,
        )
        ar_sample_sets[label] = param_samples.ar

    min_ar = 1
    max_ar = -1
    print(f"TRUE: {true_param.ar}")
    for label, ar_set in ar_sample_sets.items():
        q05, q95 = jnp.quantile(ar_set, jnp.array([0.05, 0.95]))
        print(f"{label}: {jnp.mean(ar_set):.2f} ({q05:.2f}, {q95:.2f})")
        min_ar = min(min_ar, jnp.min(ar_set))
        max_ar = max(max_ar, jnp.max(ar_set))

    plt.figure(figsize=(6, 3))
    bins = jnp.linspace(min_ar, max_ar, 100)
    for label, ar_set in ar_sample_sets.items():
        plt.hist(ar_set, bins=bins, density=True, alpha=0.5, label=label)

    plt.axvline(true_params.ar, color="k", linestyle="--", label="true")
    plt.xlabel("ar parameter")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()
