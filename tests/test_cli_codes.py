from seqjax.cli.cli import build_inference_config
from seqjax.inference.optimization.registry import AdamOpt, CosineOpt
from seqjax.inference.vi.registry import (
    BufferedVIConfig,
        MAFLatentApproximation,
    MAFParameterApproximation,
    MultivariateNormalParameterApproximation,
    PassthroughEmbedder,
    StructuredPrecisionLatentApproximation,
)


def test_buffer_vi_prior_training_defaults_to_none() -> None:
    config = build_inference_config("buffer-vi", [])
    assert isinstance(config.config, BufferedVIConfig)
    assert config.config.prior_training_optimization is None


def test_buffer_vi_prior_training_adam_code() -> None:
    config = build_inference_config("buffer-vi", ["PR.ADAM", "PR.ADAM.LR-1e-2"])
    assert isinstance(config.config, BufferedVIConfig)
    assert isinstance(config.config.prior_training_optimization, AdamOpt)
    assert config.config.prior_training_optimization.lr == 1e-2


def test_buffer_vi_optimization_cosine_code() -> None:
    config = build_inference_config(
        "buffer-vi",
        [
            "OPT.COS",
            "OPT.COS.WARM-50",
            "OPT.COS.DEC-400",
            "OPT.COS.PEAK-2e-2",
            "OPT.COS.END-1e-4",
            "OPT.COS.MAXS-500",
        ],
    )
    assert isinstance(config.config, BufferedVIConfig)
    assert isinstance(config.config.optimization, CosineOpt)
    assert config.config.optimization.warmup_steps == 50
    assert config.config.optimization.decay_steps == 400
    assert config.config.optimization.peak_lr == 2e-2
    assert config.config.optimization.end_lr == 1e-4
    assert config.config.optimization.total_steps == 500


def test_buffer_vi_embedder_passthrough_code() -> None:
    config = build_inference_config("buffer-vi", ["EMB.PS"])
    assert isinstance(config.config, BufferedVIConfig)
    assert isinstance(config.config.embedder, PassthroughEmbedder)


def test_buffer_vi_parameter_approximation_registry_options() -> None:
    maf_config = build_inference_config(
        "buffer-vi",
        ["PAX.MAF", "PAX.MAF.W-40", "PAX.MAF.D-3"],
    )
    assert isinstance(maf_config.config, BufferedVIConfig)
    assert isinstance(maf_config.config.parameter_approximation, MAFParameterApproximation)
    assert maf_config.config.parameter_approximation.nn_width == 40
    assert maf_config.config.parameter_approximation.nn_depth == 3

    mvn_config = build_inference_config("buffer-vi", ["PAX.MVN", "PAX.MVN.J-1e-5"])
    assert isinstance(mvn_config.config, BufferedVIConfig)
    assert isinstance(
        mvn_config.config.parameter_approximation,
        MultivariateNormalParameterApproximation,
    )
    assert mvn_config.config.parameter_approximation.diag_jitter == 1e-5


def test_buffer_vi_latent_approximation_structured_code() -> None:
    config = build_inference_config("buffer-vi", ["LAX.STR", "LAX.STR.W-48", "LAX.STR.D-4"])
    assert isinstance(config.config, BufferedVIConfig)
    assert isinstance(config.config.latent_approximation, StructuredPrecisionLatentApproximation)
    assert config.config.latent_approximation.nn_width == 48
    assert config.config.latent_approximation.nn_depth == 4


def test_buffer_vi_latent_approximation_maf_still_supported() -> None:
    config = build_inference_config("buffer-vi", ["LAX.MAF", "LAX.MAF.W-30", "LAX.MAF.D-2"])
    assert isinstance(config.config, BufferedVIConfig)
    assert isinstance(config.config.latent_approximation, MAFLatentApproximation)
    assert config.config.latent_approximation.nn_width == 30
