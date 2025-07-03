import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import simulate
from seqjax.model.stochastic_vol import (
    SkewStochasticVol,
    LogVolWithSkew,
    TimeIncrement,
)


def simulate_path(skew: float, steps: int = 10000, seed: int = 0):
    """Simulate a path from the skew stochastic volatility model."""
    params = LogVolWithSkew(
        std_log_vol=jnp.array(3.2),
        mean_reversion=jnp.array(12.0),
        long_term_vol=jnp.array(0.16),
        skew=jnp.array(skew),
    )

    dt = jnp.array(1.0 / (256 * 8))
    cond = TimeIncrement(dt * jnp.ones(steps))

    key = jrandom.key(seed)
    latent, obs = simulate.simulate(
        key,
        SkewStochasticVol,
        cond,
        params,
        sequence_length=steps,
    )

    returns = jnp.log(obs.underlying[1:]) - jnp.log(obs.underlying[:-1])
    realised_vol = jnp.sqrt(jnp.mean(returns**2) / dt)
    return returns, realised_vol


if __name__ == "__main__":
    steps = 2000
    betas = [0.0, -0.8]
    fig, (ax_hist, ax_vol) = plt.subplots(1, 2, figsize=(10, 4))

    for beta in betas:
        r, rv = simulate_path(beta, steps)
        r_np = jnp.asarray(r)
        ax_hist.hist(r_np, bins=50, alpha=0.5, density=True, label=f"beta={beta}")
        ax_vol.bar(beta, rv, width=0.1)

    ax_hist.set_title("Return distribution")
    ax_hist.legend()
    ax_vol.set_title("Realised volatility")
    ax_vol.set_xlabel("beta")
    plt.tight_layout()
    plt.show()
