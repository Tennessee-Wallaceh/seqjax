import jax.random as jrandom
import jax.numpy as jnp

from seqjax.model import simulate
from seqjax.model.ar import AR1Target, ARParameters, NoisyEmission
from seqjax.inference.vi.autoregressive.autoregressive_vi import RandomAutoregressor
from seqjax.inference.vi import Variational, ParameterModel, MeanField, Constraint, Identity
from seqjax.inference.embedder import PassThroughEmbedder


def test_vi_sample_shape() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    params = ARParameters()
    _, obs, _, _ = simulate.simulate(key, target, None, params, sequence_length=5)

    # add a single context axis
    obs_batched = NoisyEmission(y=jnp.expand_dims(obs.y, 0))

    embedder = PassThroughEmbedder(sample_length=5, prev_window=0, post_window=0, y_dimension=1)
    sampler = RandomAutoregressor(
        sample_length=5,
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
        target_particle=target.particle_type,
    )

    keys = jnp.array([jrandom.PRNGKey(1)])
    x, log_q_x, theta, log_q_theta = vi.sample_and_log_prob(obs_batched, keys, 2)

    assert x.x.shape == (1, 2, 5)
    assert log_q_x.shape == (1, 2)
    assert theta.ar.shape == (1, 2)
    assert log_q_theta.shape == (1, 2)
