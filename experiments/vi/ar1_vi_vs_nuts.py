import jax
import jax.numpy as jnp
import jax.random as jrandom
import blackjax
import equinox as eqx

from seqjax.model import simulate, evaluate
from seqjax.model.ar import AR1Target, ARParameters, NoisyEmission
from seqjax.inference.vi.autoregressive.autoregressive_vi import RandomAutoregressor
from seqjax.inference.vi import Variational, ParameterModel, MeanField, Constraint, Identity
from seqjax.inference.embedder import PassThroughEmbedder


if __name__ == "__main__":
    true_params = ARParameters(ar=jnp.array(0.8))
    key = jrandom.PRNGKey(0)
    _, obs, _, _ = simulate.simulate(key, AR1Target(), None, true_params, sequence_length=25)
    obs_batched = NoisyEmission(y=jnp.expand_dims(obs.y, 0))

    embedder = PassThroughEmbedder(sample_length=25, prev_window=0, post_window=0, y_dimension=1)
    sampler = RandomAutoregressor(
        sample_length=25,
        x_dim=1,
        context_dim=embedder.context_dimension,
        parameter_dim=1,
        lag_order=1,
    )
    parameter_model = ParameterModel(
        dim=1,
        base_flow=MeanField(1),
        constraint=Constraint(dim=1, dim_ix=[[0]], bijections=[Identity()]),
        parameter_map=["ar"],
        target_parameters=ARParameters,
    )
    vi = Variational(
        sampler=sampler,
        parameter_model=parameter_model,
        embedder=embedder,
        target_particle=AR1Target.particle_type,
    )

    log_prob_joint = evaluate.get_log_prob_joint_for_target(AR1Target())

    def elbo(v: Variational, k: jax.Array) -> jax.Array:
        x, log_q_x, theta, log_q_theta = v.sample_and_log_prob(
            obs_batched, jnp.array([k]), 1
        )
        params = ARParameters(
            ar=theta.ar.squeeze(),
            observation_std=true_params.observation_std,
            transition_std=true_params.transition_std,
        )
        x_path = AR1Target.particle_type(x=x.x.squeeze((0, 1)))
        logp = log_prob_joint(x_path, obs, None, params)
        return logp - log_q_x.squeeze() - log_q_theta.squeeze()

    grad_elbo = eqx.filter_grad(lambda v, k: -elbo(v, k))

    lr = 0.005
    for _ in range(100):
        key, sub = jrandom.split(key)
        grads = grad_elbo(vi, sub)
        vi = eqx.apply_updates(vi, jax.tree_util.tree_map(lambda g: -lr * g, grads))

    # VI samples
    key, sub = jrandom.split(key)
    theta_vi, _ = vi.sample_theta_and_log_prob(jnp.array([sub]), 100)
    vi_est = jnp.mean(theta_vi.ar)

    # NUTS
    logdensity = lambda state: log_prob_joint(
        AR1Target.particle_type.from_array(state[0]),
        obs,
        None,
        ARParameters(
            ar=state[1],
            observation_std=true_params.observation_std,
            transition_std=true_params.transition_std,
        ),
    )

    init_state = (
        vi.sampler.sample_single_path(
            key, jnp.array([0.0]), embedder.embed(jnp.expand_dims(obs.y, -1))
        )[0],
        jnp.array(0.0),
    )
    nuts = blackjax.nuts(
        logdensity, step_size=0.005, inverse_mass_matrix=jnp.ones(26)
    )
    state = nuts.init(init_state)

    keys = jrandom.split(key, 60)
    for k in keys[:10]:
        state, _ = nuts.step(k, state)
    ar_samples = []
    latent_samples = []
    for k in keys[10:]:
        state, _ = nuts.step(k, state)
        latent_samples.append(state.position[0])
        ar_samples.append(state.position[1])
    latent_samples = jnp.stack(latent_samples)
    ar_samples = jnp.asarray(ar_samples)

    print("VI mean ar:", float(vi_est))
    print("NUTS mean ar:", float(jnp.mean(ar_samples)))

    import matplotlib.pyplot as plt
    import jax.scipy.stats as jstats

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    # density histogram for theta
    axes[0].hist(ar_samples, bins=20, density=True, alpha=0.5, label="NUTS")

    loc = vi.parameter_model.base_flow.loc[0]
    scale = 1e-3 + jax.nn.softplus(vi.parameter_model.base_flow._unc_scale[0])
    grid = jnp.linspace(loc - 4 * scale, loc + 4 * scale, 200)
    axes[0].plot(grid, jstats.norm.pdf(grid, loc=loc, scale=scale), label="VI")
    axes[0].set_title("AR parameter distribution")
    axes[0].legend()

    # latent trajectory comparison
    latent_vi, _, _, _ = vi.sample_and_log_prob(obs_batched, jnp.array([key]), 5)
    latent_vi = latent_vi.x.squeeze(0)[:, :, 0]
    latent_nuts = latent_samples[:5, :, 0]
    for i in range(5):
        axes[1].plot(latent_nuts[i], color="C0", alpha=0.6, label="NUTS" if i == 0 else None)
        axes[1].plot(latent_vi[i], color="C1", alpha=0.6, label="VI" if i == 0 else None)
    axes[1].set_title("Latent state samples")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
