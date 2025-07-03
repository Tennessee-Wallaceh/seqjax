import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model.sir import SIRModel, SIRParameters
from seqjax import BootstrapParticleFilter, AuxiliaryParticleFilter, simulate
from seqjax.inference.particlefilter import (
    run_filter,
    vmapped_run_filter,
    current_particle_quantiles,
    current_particle_variance,
)


def test_ar1_bootstrap_filter_runs() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )
    filter_key = jrandom.PRNGKey(1)
    bpf = BootstrapParticleFilter(target, num_particles=10)
    log_w, particles, log_mp, ess, rec = run_filter(
        bpf,
        filter_key,
        parameters,
        observations,
        initial_conditions=(None,),
    )

    assert log_w.shape == (bpf.num_particles,)
    assert log_mp.shape == (observations.y.shape[0],)
    assert ess.shape == (observations.y.shape[0],)
    assert rec == ()


def test_run_filter_requires_initial_conditions() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )
    filter_key = jrandom.PRNGKey(1)
    bpf = BootstrapParticleFilter(target, num_particles=10)

    with pytest.raises(ValueError):
        run_filter(bpf, filter_key, parameters, observations)


def test_particle_recorders_shapes() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )
    filter_key = jrandom.PRNGKey(1)
    bpf = BootstrapParticleFilter(target, num_particles=20)

    quant_rec = current_particle_quantiles(lambda p: p.x)
    var_rec = current_particle_variance(lambda p: p.x)

    log_w, _, _, _, (quant_hist, var_hist) = run_filter(
        bpf,
        filter_key,
        parameters,
        observations,
        initial_conditions=(None,),
        recorders=(quant_rec, var_rec),
    )

    seq_len = observations.y.shape[0]
    assert quant_hist.shape == (seq_len, 2)
    assert var_hist.shape == (seq_len,)


def test_ar1_auxiliary_filter_runs() -> None:
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )
    filter_key = jrandom.PRNGKey(1)
    apf = AuxiliaryParticleFilter(target, num_particles=10)
    log_w, particles, log_mp, ess, _ = run_filter(
        apf,
        filter_key,
        parameters,
        observations,
        initial_conditions=(None,),
    )

    assert log_w.shape == (apf.num_particles,)
    assert ess.shape == (observations.y.shape[0],)


def test_sir_bootstrap_filter_runs() -> None:
    key = jrandom.PRNGKey(0)
    target = SIRModel()
    parameters = SIRParameters(
        infection_rate=jnp.array(0.1),
        recovery_rate=jnp.array(0.05),
        population=jnp.array(100.0),
    )

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )
    filter_key = jrandom.PRNGKey(1)
    bpf = BootstrapParticleFilter(target, num_particles=10)
    log_w, _, _, ess, _ = run_filter(
        bpf,
        filter_key,
        parameters,
        observations,
        initial_conditions=(None,),
    )

    assert log_w.shape == (bpf.num_particles,)
    assert ess.shape == (observations.new_cases.shape[0],)


def test_vmapped_run_filter_shapes() -> None:
    batch = 3
    key = jrandom.PRNGKey(0)
    target = AR1Target()
    parameters = ARParameters()

    _, observations, _, _ = simulate.simulate(
        key, target, None, parameters, sequence_length=5
    )

    bpf = BootstrapParticleFilter(target, num_particles=4)

    keys = jrandom.split(jrandom.PRNGKey(1), batch)

    batched_params = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (batch,) + jnp.shape(x)),
        parameters,
    )
    batched_obs = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (batch,) + jnp.shape(x)),
        observations,
    )

    log_w, _, log_mp, _, _ = vmapped_run_filter(
        bpf,
        keys,
        batched_params,
        batched_obs,
        initial_conditions=(None,),
    )

    assert log_w.shape == (batch, bpf.num_particles)
    assert log_mp.shape == (batch, observations.y.shape[0])
