from typer.testing import CliRunner

from seqjax.cli.cli import app, build_latent_inference_config
from seqjax.inference.vi.registry import (
    AutoregressiveLatentApproximation,
    ConvNFLatentApproximation,
    MAFLatentApproximation,
    StructuredPrecisionLatentApproximation,
)


def test_cli_help_lists_latent_fit_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "latent-fit" in result.stdout


def test_build_latent_inference_config_autoregressive_codes() -> None:
    config = build_latent_inference_config(["LAX.SEQ", "LAX.SEQ.W-40", "LAX.SEQ.D-3", "LAX.SEQ.L-2"])
    assert isinstance(config.latent_approximation, AutoregressiveLatentApproximation)
    assert config.latent_approximation.nn_width == 40
    assert config.latent_approximation.nn_depth == 3
    assert config.latent_approximation.lag_order == 2


def test_build_latent_inference_config_maf_codes() -> None:
    config = build_latent_inference_config(
        ["LAX.MAF", "LAX.MAF.W-30", "LAX.MAF.D-2", "LAX.MAF.FL-4", "LAX.MAF.BL-0.5", "LAX.MAF.BS-1.5"],
    )
    assert isinstance(config.latent_approximation, MAFLatentApproximation)
    assert config.latent_approximation.nn_width == 30
    assert config.latent_approximation.nn_depth == 2
    assert config.latent_approximation.flow_layers == 4
    assert config.latent_approximation.base_loc == 0.5
    assert config.latent_approximation.base_scale == 1.5


def test_build_latent_inference_config_structured_codes() -> None:
    config = build_latent_inference_config(["LAX.STR", "LAX.STR.W-24", "LAX.STR.D-5"])
    assert isinstance(config.latent_approximation, StructuredPrecisionLatentApproximation)
    assert config.latent_approximation.nn_width == 24
    assert config.latent_approximation.nn_depth == 5


def test_build_latent_inference_config_conv_flow_codes() -> None:
    config = build_latent_inference_config(["LAX.CNF", "LAX.CNF.W-22", "LAX.CNF.D-4", "LAX.CNF.K-7", "LAX.CNF.FL-3"])
    assert isinstance(config.latent_approximation, ConvNFLatentApproximation)
    assert config.latent_approximation.nn_width == 22
    assert config.latent_approximation.nn_depth == 4
    assert config.latent_approximation.kernel_size == 7
    assert config.latent_approximation.flow_layers == 3


def test_build_latent_inference_config_reads_opt_codes() -> None:
    config = build_latent_inference_config(["OPT.ADAM", "OPT.ADAM.LR-2e-3"])
    assert config.optimization.lr == 2e-3
