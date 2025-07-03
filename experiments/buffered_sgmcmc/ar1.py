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
    _, obs, _, _ = simulate.simulate(key, AR1Target(), None, true_params, sequence_length=50)

    pf = BootstrapParticleFilter(AR1Target(), num_particles=128)
    config = BufferedSGLDConfig(
        step_size=1e-2,
        num_iters=200,
        buffer_size=1,
        batch_size=10,
        particle_filter=pf,
        parameter_prior=HalfCauchyStds(),
    )

    init_params = ARParameters(ar=jnp.array(0.0))
    samples = run_buffered_sgld(AR1Target(), jrandom.PRNGKey(1), init_params, obs, config=config)

    ar_samples = jnp.asarray(samples.ar)
    print("True AR parameter:", float(true_params.ar))
    print("Mean SGLD estimate:", float(jnp.mean(ar_samples)))

    plt.plot(ar_samples, label="sampled ar")
    plt.axhline(true_params.ar, color="r", linestyle="--", label="true")
    plt.xlabel("iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()
