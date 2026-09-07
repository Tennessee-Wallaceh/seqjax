import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from seqjax.inference import ObservationDataset
from seqjax.model import ar
from seqjax.model.condition import normalize_condition_path
from seqjax.model.simulate import simulate
from seqjax.model.stochastic_vol import time_variable_var
from seqjax.model.typing import NoCondition


def test_no_condition_preserves_axes_through_jax_transformations() -> None:
    conditions = NoCondition.for_batch_shape((2, 3))

    mapped = jax.vmap(lambda condition: condition.ravel())(conditions)
    _, scanned = jax.lax.scan(
        lambda state, condition: (state, condition.ravel()),
        None,
        conditions,
    )

    assert conditions.flat_dim == 0
    assert conditions.batch_shape == (2, 3)
    assert mapped.shape == (2, 3, 0)
    assert scanned.shape == (2, 3, 0)


def test_public_boundaries_materialize_no_condition_axes() -> None:
    latent_path, observation_path = simulate(
        jrandom.key(0),
        ar.ar_model,
        ar.ARParameters(),
        sequence_length=4,
    )
    dataset = ObservationDataset.from_single_sequence(observation_path)

    assert latent_path.batch_shape == (4,)
    assert isinstance(dataset.conditions, NoCondition)
    assert dataset.conditions.batch_shape == (1, 4)
    assert dataset.sequence(0)[1].batch_shape == (4,)


def test_missing_required_condition_is_rejected() -> None:
    model = time_variable_var.SimpleStochasticVar()

    with pytest.raises(ValueError, match="requires a condition path"):
        normalize_condition_path(model, None, (4,))


def test_real_condition_for_unconditional_model_is_rejected() -> None:
    condition = time_variable_var.TimeStepCondition(timestep=jnp.ones((4,)))

    with pytest.raises(ValueError, match="is unconditional"):
        normalize_condition_path(ar.ar_model, condition, (4,))
