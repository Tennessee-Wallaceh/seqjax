from __future__ import annotations

import pytest
import jax.numpy as jnp
import jax.random as jrandom

from seqjax.model import evaluate, simulate
from seqjax.model import linear_gaussian as lg
from seqjax.model import linear_gaussian_bayesian as lgb


def test_dynamic_packable_templates_follow_class_arguments() -> None:
    state_cls = lg.make_vector_state_cls(3)
    observation_cls = lg.make_vector_observation_cls(3)
    parameter_cls = lg.make_lgssm_parameters_cls(3)

    assert state_cls.dim == 3
    assert observation_cls.dim == 3
    assert parameter_cls.dim == 3
    assert state_cls._shape_template["x"].shape == (3,)
    assert observation_cls._shape_template["y"].shape == (3,)
    assert parameter_cls._shape_template["transition_matrix"].shape == (3, 3)


def test_lgssm_factory_supports_custom_dimension() -> None:
    model = lg.lgssm(3)
    parameters = model.parameter_cls()
    condition = lg.NoCondition()

    latents, observations = simulate.simulate(
        jrandom.PRNGKey(0),
        model,
        parameters=parameters,
        condition=condition,
        sequence_length=4,
    )

    assert model.dim == 3
    assert latents.x.shape[-1] == 3
    assert observations.y.shape[-1] == 3
    assert jnp.isfinite(
        evaluate.log_prob_joint(
            model,
            latents,
            observations,
            parameters=parameters,
            condition=condition,
        )
    )


def test_bayesian_lgssm_respects_dimension_hyperparameter() -> None:
    hyperparameters = lgb.LGSSMHyperParameters(dim=4)
    bayesian_model = lgb.lgssm_full(hyperparameters)

    assert bayesian_model.target.parameter_cls.dim == 4
    assert bayesian_model.parameterization.inference_parameter_cls.dim == 4

    unc_parameters = bayesian_model.parameterization.sample(jrandom.PRNGKey(1))
    model_parameters = bayesian_model.parameterization.to_model_parameters(unc_parameters)
    recovered = bayesian_model.parameterization.from_model_parameters(model_parameters)

    assert unc_parameters.unc_transition_diag.shape == (4,)
    assert unc_parameters.emission_strictly_lower.shape == (6,)
    assert model_parameters.transition_matrix.shape == (4, 4)
    assert recovered.emission_log_diag.shape == (4,)
    assert jnp.isfinite(bayesian_model.parameterization.log_prob(unc_parameters))


@pytest.mark.parametrize("bad_dim", [0, -2])
def test_invalid_dimensions_raise_helpful_errors(bad_dim: int) -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        lg.lgssm(bad_dim)

    with pytest.raises(ValueError, match="dim must be positive"):
        lgb.LGSSMHyperParameters(dim=bad_dim)
