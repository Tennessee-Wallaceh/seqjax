"""Tests covering all registered inference methods."""

from collections.abc import Iterator
from functools import partial

import jax.numpy as jnp
import jax.random as jrandom
import pytest

from _pytest.mark.structures import ParameterSet

from tqdm import auto as tqdm_auto
from tqdm import notebook as tqdm_notebook

from seqjax.inference import (
    mcmc,
    particlefilter,
    pmcmc,
    registry as inference_registry,
    sgld,
    vi,
)
from seqjax.model import ar, registry as model_registry, simulate


class _DummyTqdmIterator(Iterator[int]):
    """Minimal iterator with ``set_postfix`` to satisfy ``tqdm`` usage in tests."""

    def __init__(self, n: int) -> None:
        self._iter = iter(range(int(n)))

    def __iter__(self) -> "_DummyTqdmIterator":
        return self

    def __next__(self) -> int:
        return next(self._iter)

    def set_postfix(self, *_args, **_kwargs) -> None:  # pragma: no cover - behaviourless stub
        """Match the API used in :mod:`seqjax.inference.vi.train`."""

@pytest.fixture(autouse=True)
def _stub_tqdm_trange(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``tqdm`` progress bars with a deterministic iterator."""

    def _trange(n: int, *_args, **_kwargs) -> _DummyTqdmIterator:
        return _DummyTqdmIterator(n)

    monkeypatch.setattr(tqdm_auto, "trange", _trange)
    monkeypatch.setattr(tqdm_notebook, "trange", _trange)


INFERENCE_TEST_SETUPS: dict[str, tuple[object, int]] = {
    "NUTS": (
        mcmc.NUTSConfig(step_size=1e-1, num_adaptation=5, num_warmup=5, num_chains=1),
        10,
    ),
    "buffer-vi": (
        vi.registry.BufferedVIConfig(
            optimization=vi.run.AdamOpt(lr=1e-2, total_steps=2),
            buffer_length=2,
            batch_length=4,
            observations_per_step=1,
            samples_per_context=1,
            control_variate=False,
            pre_training_optimization=vi.run.AdamOpt(lr=1e-2, total_steps=2),
            latent_approximation=vi.run.AutoregressiveLatentApproximation(
                nn_width=4,
                nn_depth=1,
            ),
            embedder=vi.registry.BiRNNEmbedder,
        ),
        200,
    ),
}
