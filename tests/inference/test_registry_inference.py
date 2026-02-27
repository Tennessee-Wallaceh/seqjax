"""Tests covering all registered inference methods."""

from collections.abc import Iterator

import pytest

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
        mcmc.NUTSConfig(
            step_size=1e-1,
            num_adaptation=5,
            num_warmup=5,
            num_steps=10,
            num_chains=1,
        ),
        10,
    ),
    "buffer-vi": (
        vi.registry.BufferedVIConfig(
            optimization=vi.run.AdamOpt(lr=1e-2, total_steps=2),
            buffer_length=2,
            batch_length=4,
            num_context_per_sequence=1,
            samples_per_context=1,
            pre_training_optimization=vi.run.AdamOpt(lr=1e-2, total_steps=2),
            latent_approximation=vi.run.AutoregressiveLatentApproximation(
                nn_width=4,
                nn_depth=1,
            ),
            embedder=vi.registry.BiRNNEmbedder(),
        ),
        200,
    ),
    "full-vi": (
        vi.registry.FullVIConfig(
            optimization=vi.run.AdamOpt(lr=1e-2, total_steps=2),
            parameter_field_bijections={},
            embedder=vi.registry.BiRNNEmbedder(),
            samples_per_context=1,
            pre_training_optimization=vi.run.AdamOpt(lr=1e-2, total_steps=1),
            prior_training_optimization=vi.run.AdamOpt(lr=1e-2, total_steps=1),
            latent_approximation=vi.registry.AutoregressiveLatentApproximation(
                nn_width=4,
                nn_depth=1,
            ),
            parameter_approximation=vi.registry.MeanFieldParameterApproximation(),
        ),
        200,
    ),
    "particle-mcmc": (
        pmcmc.ParticleMCMCConfig(
            particle_filter_config=particlefilter.registry.BootstrapFilterConfig(
                resample="multinomial",
                num_particles=2,
            ),
            mcmc_config=mcmc.RandomWalkConfig(step_size=0.1),
            num_steps=2,
        ),
        10,
    ),
    "full-sgld": (
        sgld.SGLDConfig(
            particle_filter_config=particlefilter.registry.BootstrapFilterConfig(
                resample="multinomial",
                num_particles=2,
            ),
            step_size=1e-3,
            num_steps=2,
        ),
        10,
    ),
    "buffer-sgld": (
        sgld.BufferedSGLDConfig(
            particle_filter_config=particlefilter.registry.BootstrapFilterConfig(
                resample="multinomial",
                num_particles=2,
            ),
            step_size=1e-3,
            num_steps=2,
            buffer_length=2,
            batch_length=4,
        ),
        10,
    ),
}


def _iter_registry_entries():
    for label, spec in inference_registry.inference_registry.items():
        if label not in INFERENCE_TEST_SETUPS:
            raise AssertionError(f"Missing inference test setup for {label!r}")
        config, test_samples = INFERENCE_TEST_SETUPS[label]
        yield pytest.param(label, spec, config, test_samples, id=label)


@pytest.mark.parametrize(
    "label, spec, config, test_samples",
    list(_iter_registry_entries()),
)
def test_registry_inference_configs_cover_registry(
    label: str,
    spec: inference_registry.InferenceSpec,
    config: object,
    test_samples: int,
) -> None:
    assert label == spec.label
    assert isinstance(config, spec.config_cls)
    assert isinstance(spec.name_fn(label, config), str)
    assert test_samples > 0
