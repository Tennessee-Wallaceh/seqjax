import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from seqjax.inference.particlefilter import run_filter
from seqjax.inference.particlefilter import registry as pf_registry
from seqjax.model import linear_gaussian as lg


@pytest.mark.parametrize("resample_kind", ["multinomial", "systematic", "none"])
def test_resampler_preserves_input_mass_in_expectation(resample_kind) -> None:
    start_log_w = jax.nn.log_softmax(jnp.linspace(-1.0, 1.0, 8))
    resampler = pf_registry.resample_registry[resample_kind]

    def resample(key):
        result = resampler(key, start_log_w, 8)
        return jnp.zeros(8).at[result.indices].add(jnp.exp(result.log_weights))

    empirical_mass = jax.vmap(resample)(jrandom.split(jrandom.key(0), 2048)).mean(
        axis=0
    )
    assert jnp.allclose(empirical_mass, jnp.exp(start_log_w), atol=0.02)


@pytest.mark.parametrize("filter_kind", ["bootstrap", "auxiliary"])
@pytest.mark.parametrize("resample_kind", ["multinomial", "systematic", "none"])
def test_population_proposal_contract(filter_kind, resample_kind) -> None:
    model = lg.lgssm(1)
    parameters = model.parameter_cls(
        emission_noise_cholesky=jnp.array([[2.0]]),
    )
    config_cls = (
        pf_registry.BootstrapFilterConfig
        if filter_kind == "bootstrap"
        else pf_registry.AuxiliaryFilterConfig
    )
    particle_filter = pf_registry.build_filter(
        model,
        config_cls(resample=resample_kind, num_particles=8),
    )
    prior = jax.vmap(model.prior_sample, in_axes=(0, None))(
        jrandom.split(jrandom.key(1), 8), parameters
    )
    context = particle_filter.filter_context(prior.values, ())
    start_log_w = jax.nn.log_softmax(jnp.linspace(-1.0, 1.0, 8))
    observation = model.observation_cls(y=jnp.array([0.4]))

    result = particle_filter.proposal(
        jrandom.key(2),
        start_log_w,
        context,
        observation,
        parameters,
        model.condition_cls(),
        8,
    )

    assert result.log_weight.shape == (8,)
    assert result.ancestor_indices.shape == (8,)
    assert result.particles.length == model.latent_context_length
    assert jnp.all(jnp.isfinite(result.log_weight))
    assert jnp.isfinite(result.log_normalizer_increment)
    assert jnp.allclose(jax.scipy.special.logsumexp(result.log_weight), 0.0, atol=1e-6)


def test_auxiliary_filter_preserves_history_and_finite_weights() -> None:
    model = lg.lgssm(1)
    parameters = model.parameter_cls()
    observations = model.observation_cls(y=jnp.zeros((4, 1)))
    particle_filter = pf_registry.build_filter(
        model,
        pf_registry.AuxiliaryFilterConfig(
            resample="systematic",
            num_particles=16,
        ),
    )

    log_w, context, _ = run_filter(
        jrandom.key(3), particle_filter, parameters, observations
    )

    assert context.length == model.latent_context_length
    assert jnp.all(jnp.isfinite(log_w))


def test_auxiliary_filter_zero_emission_matrix_has_exact_increments() -> None:
    model = lg.lgssm(1)
    parameters = model.parameter_cls(
        emission_matrix=jnp.zeros((1, 1)),
        emission_noise_cholesky=jnp.array([[0.7]]),
    )
    observation_values = jnp.array([[-0.3], [0.1], [1.2], [-0.7]])
    observations = model.observation_cls(y=observation_values)
    particle_filter = pf_registry.build_filter(
        model,
        pf_registry.AuxiliaryFilterConfig(
            resample="multinomial",
            num_particles=64,
        ),
    )

    log_w, _, (log_z_inc,) = run_filter(
        jrandom.key(4),
        particle_filter,
        parameters,
        observations,
        recorders=(lambda data: data.log_z_inc,),
    )
    scale = parameters.emission_noise_cholesky[0, 0]
    expected = (
        -0.5 * jnp.square(observation_values[:, 0] / scale)
        - jnp.log(scale)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )

    assert jnp.allclose(log_z_inc, expected, atol=1e-5)
    assert jnp.allclose(jnp.exp(log_w), jnp.full((64,), 1.0 / 64), atol=1e-6)


def test_no_resample_rejects_particle_count_change() -> None:
    from seqjax.inference.particlefilter import no_resample

    with pytest.raises(ValueError, match="requested output count"):
        no_resample(jrandom.key(5), jnp.zeros(3), 2)


def test_build_filter_rejects_unknown_config_type() -> None:
    with pytest.raises(TypeError, match="Unsupported particle-filter configuration"):
        pf_registry.build_filter(lg.lgssm(1), object())
