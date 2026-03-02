import jax.random as jrandom
import pytest

from seqjax.inference import ObservationDataset
from seqjax.inference.vi.base import _sample_sequence_minibatch
from seqjax.model import registry as model_registry
from seqjax.model import simulate
import seqjax.model.typing as seqjtyping


def _build_single_sequence_dataset(sequence_length: int = 8) -> ObservationDataset:
    target = model_registry.sequential_models["ar"]
    params = model_registry.parameter_settings["ar"]["base"]
    _, observation_path = simulate.simulate(
        jrandom.PRNGKey(0),
        target,
        params,
        sequence_length=sequence_length,
        condition=seqjtyping.NoCondition(),
    )
    return ObservationDataset.from_single_sequence(
        observation_path=observation_path,
        condition_path=seqjtyping.NoCondition(),
    )


def test_sequence_minibatch_above_dataset_size_rejects_third_argument() -> None:
    dataset = _build_single_sequence_dataset()
    with pytest.raises(
        ValueError,
        match="num_sequence_minibatch cannot exceed dataset.num_sequences",
    ):
        _sample_sequence_minibatch(
            dataset,
            jrandom.PRNGKey(1),
            2,
        )


def test_sequence_minibatch_nonpositive_rejects_third_argument() -> None:
    dataset = _build_single_sequence_dataset()
    with pytest.raises(ValueError, match="num_sequence_minibatch must be positive"):
        _sample_sequence_minibatch(
            dataset,
            jrandom.PRNGKey(1),
            0,
        )
