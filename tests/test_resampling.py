import jax
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.particlefilter import (
    gumbel_resample_from_log_weights,
    conditional_resample,
    current_particle_mean,
    current_particle_quantiles,
    current_particle_variance,
    systematic_resample_from_log_weights,
    stratified_resample_from_log_weights,
    multinomial_resample_from_log_weights,
)


def test_gumbel_resample_from_log_weights_deterministic() -> None:
    key = jrandom.PRNGKey(0)
    log_w = jnp.log(jnp.array([0.1, 0.2, 0.7]))
    particles = (jnp.arange(3), jnp.arange(3) * 10)

    resampled, new_log_w, anc = gumbel_resample_from_log_weights(key, log_w, particles, 0.0)

    gumbels = -jnp.log(-jnp.log(jrandom.uniform(key, (log_w.shape[0], log_w.shape[0]))))
    idx = jnp.argmax(log_w + gumbels, axis=1)
    expected = tuple(jnp.take(p, idx, axis=0) for p in particles)
    expected_log_w = jnp.full_like(log_w, -jnp.log(log_w.shape[0]))

    assert all(jnp.array_equal(r, e) for r, e in zip(resampled, expected))
    assert jnp.array_equal(new_log_w, expected_log_w)
    assert jnp.array_equal(anc, idx)


def test_conditional_resample_threshold() -> None:
    key = jrandom.PRNGKey(1)
    log_w = jnp.log(jnp.array([0.4, 0.6]))
    particles = (jnp.array([0, 1]),)

    # No resampling when ess_e >= threshold
    out_p, out_w, out_a = conditional_resample(
        key,
        log_w,
        particles,
        ess_e=jnp.array(0.9),
        resampler=gumbel_resample_from_log_weights,
        esse_threshold=0.5,
    )
    assert all(jnp.array_equal(o, p) for o, p in zip(out_p, particles))
    assert jnp.array_equal(out_w, log_w)
    assert jnp.array_equal(out_a, jnp.arange(log_w.shape[0]))

    # Resampling when ess_e < threshold
    res_p, res_w, res_a = conditional_resample(
        key,
        log_w,
        particles,
        ess_e=jnp.array(0.1),
        resampler=gumbel_resample_from_log_weights,
        esse_threshold=0.5,
    )
    exp_p, exp_w, exp_a = gumbel_resample_from_log_weights(key, log_w, particles, 0.1)
    assert all(jnp.array_equal(r, e) for r, e in zip(res_p, exp_p))
    assert jnp.array_equal(res_w, exp_w)
    assert jnp.array_equal(res_a, exp_a)


def test_particle_recorders_correctness() -> None:
    weights = jnp.array([0.2, 0.3, 0.5])
    particles = (jnp.array([1.0, 2.0, 4.0]),)

    mean_rec = current_particle_mean(lambda p: p)
    quant_rec = current_particle_quantiles(lambda p: p, quantiles=(0.5,))
    var_rec = current_particle_variance(lambda p: p)

    ancestors = jnp.arange(particles[0].shape[0])
    obs = jnp.array(0.0)
    cond = jnp.array(0.0)
    mean = mean_rec(
        weights,
        particles,
        ancestors,
        obs,
        cond,
        particles,
        jnp.log(weights),
        jnp.array(0.0),
        jnp.array(0.0),
    )
    quant = quant_rec(
        weights,
        particles,
        ancestors,
        obs,
        cond,
        particles,
        jnp.log(weights),
        jnp.array(0.0),
        jnp.array(0.0),
    )
    var = var_rec(
        weights,
        particles,
        ancestors,
        obs,
        cond,
        particles,
        jnp.log(weights),
        jnp.array(0.0),
        jnp.array(0.0),
    )

    exp_mean = jnp.sum(weights * particles[0])
    exp_quant = jnp.array([2.0])
    exp_var = jnp.sum(weights * (particles[0] - exp_mean) ** 2)

    assert jnp.allclose(mean, exp_mean)
    assert jnp.allclose(quant, exp_quant)
    assert jnp.allclose(var, exp_var)


def _check_statistical_resampler(resampler) -> None:
    key = jrandom.PRNGKey(0)
    log_w = jnp.log(jnp.array([0.2, 0.3, 0.5]))
    particles = (jnp.arange(3),)

    def _single(k):
        resampled, _, _ = resampler(k, log_w, particles, 0.0)
        return resampled[0]

    keys = jrandom.split(key, 5000)
    samples = jax.vmap(_single)(keys)
    flat = samples.reshape(-1)
    counts = jnp.bincount(flat, length=3) / flat.shape[0]
    assert jnp.allclose(counts, jnp.array([0.2, 0.3, 0.5]), atol=0.02)


def test_systematic_resampler_statistics() -> None:
    _check_statistical_resampler(systematic_resample_from_log_weights)


def test_stratified_resampler_statistics() -> None:
    _check_statistical_resampler(stratified_resample_from_log_weights)


def test_multinomial_resampler_statistics() -> None:
    _check_statistical_resampler(multinomial_resample_from_log_weights)
