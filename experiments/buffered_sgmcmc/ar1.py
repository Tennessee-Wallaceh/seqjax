import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import simulate
from seqjax.model.ar import AR1Target, ARParameters, HalfCauchyStds
from seqjax import (
    BootstrapParticleFilter,
    BufferedSGLDConfig,
    run_buffered_sgld,
)


if __name__ == "__main__":
    true_params = ARParameters(ar=jnp.array(0.8))
    key = jrandom.PRNGKey(0)
    _, obs, _, _ = simulate.simulate(
        key, AR1Target(), None, true_params, sequence_length=50
    )

    pf = BootstrapParticleFilter(AR1Target(), num_particles=256)
    config = BufferedSGLDConfig(
        step_size=ARParameters(
            ar=jnp.array(5e-3), observation_std=0.0, transition_std=0.0
        ),
        num_iters=5000,
        buffer_size=1,
        batch_size=10,
        particle_filter=pf,
        parameter_prior=HalfCauchyStds(),
    )

    init_params = ARParameters(ar=jnp.array(0.0))
    samples = run_buffered_sgld(
        AR1Target(), jrandom.PRNGKey(1), init_params, obs, config=config
    )

    ar_samples = jnp.asarray(samples.ar)
    print("True AR parameter:", float(true_params.ar))
    print("Mean SGLD estimate:", float(jnp.mean(ar_samples)))

    q05, q95 = jnp.quantile(ar_samples, jnp.array([0.05, 0.95]))
    plt.plot(ar_samples, label="sampled ar")
    plt.axhline(true_params.ar, color="r", linestyle="--", label="true")
    plt.axhline(jnp.mean(ar_samples), color="g", linestyle="--", label="true")
    plt.axhline(q05, color="g", linestyle="-.")
    plt.axhline(q95, color="g", linestyle="-.")
    plt.xlabel("iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()
