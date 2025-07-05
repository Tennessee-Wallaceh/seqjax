import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import blackjax
import jax.random as jrandom

from seqjax.model import simulate, evaluate
from seqjax.model.ar import AR1Target, ARParameters


if __name__ == "__main__":
    target = AR1Target()
    true_params = ARParameters(
        ar=jnp.array(0.5),
        observation_std=jnp.array(0.01),
        transition_std=jnp.array(1.0),
    )
    sequence_length = 50

    key = jrandom.PRNGKey(0)
    latents, obs, _, _ = simulate.simulate(
        key, target, None, true_params, sequence_length=sequence_length
    )

    log_prob_joint = evaluate.get_log_prob_joint_for_target(target)

    def logdensity(state):
        lat, ar = state
        params = ARParameters(
            ar=ar,
            observation_std=true_params.observation_std,
            transition_std=true_params.transition_std,
        )
        return log_prob_joint(lat, obs, None, params)

    init_state = (latents, jnp.array(0.0))
    inv_mass = jnp.ones_like(jax.flatten_util.ravel_pytree(init_state)[0])
    nuts = blackjax.nuts(logdensity, step_size=0.005, inverse_mass_matrix=inv_mass)
    state = nuts.init(init_state)

    num_warmup = 10
    num_samples = 20
    keys = jrandom.split(jrandom.PRNGKey(1), num_warmup + num_samples)

    for k in keys[:num_warmup]:
        state, _ = nuts.step(k, state)

    ar_samples = []
    latent_samples = []
    for k in keys[num_warmup:]:
        state, _ = nuts.step(k, state)
        latent_samples.append(state.position[0].x)
        ar_samples.append(state.position[1])

    ar_samples = jnp.asarray(ar_samples)
    latent_samples = jnp.asarray(latent_samples)

    mean_latent = jnp.mean(latent_samples, axis=0)
    lower_latent = jnp.quantile(latent_samples, 0.05, axis=0)
    upper_latent = jnp.quantile(latent_samples, 0.95, axis=0)

    t = jnp.arange(sequence_length)
    plt.figure(figsize=(6, 4))
    plt.plot(t, latents.x, label="true latent")
    plt.plot(t, mean_latent, label="mean sample")
    plt.fill_between(t, lower_latent, upper_latent, color="gray", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.plot(ar_samples)
    plt.axhline(true_params.ar, color="r", linestyle="--", label="true")
    plt.axhline(jnp.mean(ar_samples), color="g", linestyle="--", label="mean")
    plt.xlabel("iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()

    q05, q95 = jnp.quantile(ar_samples, jnp.array([0.05, 0.95]))
    plt.figure(figsize=(6, 3))
    plt.hist(ar_samples, bins=30, density=True, alpha=0.7)
    plt.axvline(true_params.ar, color="r", linestyle="--", label="true")
    plt.axvline(q05, color="k", linestyle="-.", label="q05")
    plt.axvline(q95, color="k", linestyle="-.", label="q95")
    plt.legend()
    plt.tight_layout()
    plt.show()
