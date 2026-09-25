import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import pytest

from seqjax.inference.particlefilter import registry as pf_registry
from seqjax.inference.particlefilter.base import run_filter
from seqjax.model.stochastic_vol import jump_vol
from seqjax.model import linear_gaussian


@pytest.mark.parametrize(
    ("config_cls", "resample"),
    [
        (pf_registry.BootstrapFilterConfig, "multinomial"),
        (pf_registry.BootstrapFilterConfig, "systematic"),
        (pf_registry.BootstrapFilterConfig, "none"),
        (pf_registry.AuxiliaryFilterConfig, "multinomial"),
        (pf_registry.AuxiliaryFilterConfig, "systematic"),
        (pf_registry.AuxiliaryFilterConfig, "none"),
    ],
)
def test_proposal_weighted_ancestors_represent_declared_measure(
    config_cls,
    resample,
) -> None:
    target = linear_gaussian.lgssm(dim=1)
    parameters = target.parameter_cls(
        transition_matrix=jnp.array([[0.7]]),
        transition_noise_cholesky=jnp.array([[0.4]]),
        emission_matrix=jnp.array([[1.0]]),
        emission_noise_cholesky=jnp.array([[0.8]]),
    )
    num_particles = 8
    particle_filter = pf_registry.build_filter(
        target,
        config_cls(resample=resample, num_particles=num_particles),
    )
    context = particle_filter.filter_context(
        (target.latent_cls(x=jnp.linspace(-1.5, 1.5, num_particles)[:, None]),),
        (),
    )
    start_log_w = jnp.log(jnp.array([0.03, 0.07, 0.10, 0.15, 0.20, 0.18, 0.15, 0.12]))
    observation = target.observation_cls(y=jnp.array([0.25]))
    condition = target.condition_cls()
    keys = jax.random.split(jax.random.key(10), 512)

    results = jax.vmap(
        particle_filter.proposal.propose,
        in_axes=(0, None, None, None, None, None, None),
    )(
        keys,
        start_log_w,
        context,
        observation,
        parameters,
        condition,
        num_particles,
    )

    ancestor_indicators = jax.nn.one_hot(results.ancestor_ix, num_particles)
    represented_mass = jnp.sum(
        jnp.exp(results.resampled_log_w)[..., None] * ancestor_indicators,
        axis=1,
    )
    declared_mass = jnp.exp(results.ancestor_log_prob)
    assert jnp.allclose(
        jnp.mean(represented_mass, axis=0),
        jnp.mean(declared_mass, axis=0),
        atol=0.02,
    )

    selected_input_log_w = start_log_w[results.ancestor_ix]
    selected_ancestor_log_prob = jnp.take_along_axis(
        results.ancestor_log_prob,
        results.ancestor_ix,
        axis=1,
    )
    corrected_mass = jnp.sum(
        jnp.exp(
            results.resampled_log_w + selected_input_log_w - selected_ancestor_log_prob
        )[..., None]
        * ancestor_indicators,
        axis=1,
    )
    assert jnp.allclose(
        jnp.mean(corrected_mass, axis=0),
        jnp.exp(start_log_w),
        atol=0.02,
    )


def test_filter_preserves_model_history_through_resampling() -> None:
    posterior = jump_vol.svar_micro_contam()
    model_parameters = jump_vol.MicroContamParams(
        std_log_var=jnp.array(0.2),
        ar=jnp.array(0.8),
        long_term_log_var=jnp.array(-2.0),
        micro_ar=jnp.array(0.5),
        micro_std=jnp.array(0.1),
        contam_prob=jnp.array(0.05),
        contam_scale=jnp.array(4.0),
    )
    observations = jump_vol.LogReturnObs(log_return=jnp.zeros(3))
    particle_filter = pf_registry.build_filter(
        posterior.target,
        pf_registry.BootstrapFilterConfig(
            resample="multinomial",
            num_particles=16,
        ),
    )

    log_weights, particle_history, _ = run_filter(
        jax.random.key(0),
        particle_filter,
        model_parameters,
        observations,
    )

    assert particle_filter.target.transition_latent_order == 1
    assert particle_filter.target.emission_latent_order == 1
    assert (
        len(particle_history.particles.values)
        == particle_filter.target.latent_context_length
    )
    assert jnp.all(jnp.isfinite(log_weights))


def test_auxiliary_filter_applies_first_and_second_stage_corrections() -> None:
    target = linear_gaussian.lgssm(dim=1)
    parameters = target.parameter_cls(
        transition_matrix=jnp.array([[0.8]]),
        transition_noise_cholesky=jnp.array([[0.5]]),
        emission_matrix=jnp.array([[0.0]]),
        emission_noise_cholesky=jnp.array([[2.0]]),
    )
    observations = target.observation_cls(y=jnp.array([[0.2], [-0.4], [1.0]]))
    particle_filter = pf_registry.build_filter(
        target,
        pf_registry.AuxiliaryFilterConfig(
            resample="multinomial",
            num_particles=32,
        ),
    )

    log_weights, _, (log_z_increments,) = run_filter(
        jax.random.key(1),
        particle_filter,
        parameters,
        observations,
        recorders=(lambda data: data.log_z_inc,),
    )

    expected = jstats.norm.logpdf(observations.y[:, 0], loc=0.0, scale=2.0)
    assert jnp.allclose(log_z_increments, expected, atol=1e-6)
    assert jnp.allclose(log_weights, -jnp.log(particle_filter.num_particles))
