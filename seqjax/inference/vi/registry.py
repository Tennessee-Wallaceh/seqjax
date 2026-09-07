import typing
from dataclasses import dataclass, field
from functools import partial

import equinox as eqx
import jax.random as jrandom
import jaxtyping

import seqjax.model.typing as seqjtyping
from seqjax.inference.optimization import registry as optimization_registry
from seqjax.inference.vi import transformations, hybrid
from seqjax.inference.vi import transformed
from seqjax.inference.vi import base

from seqjax.inference.vi.embedder import EmbedderConfig, LatentContextDims, build_embedder

from seqjax.inference.vi import maf
from seqjax.inference.vi import conv_nf
from seqjax.inference.vi import autoregressive
from seqjax.inference.vi import structured
from seqjax.inference.vi.sampling import VISampleConfig, VISamplingKwargs
from seqjax.model import interface as model_interface
from seqjax.inference.particlefilter import registry as particle_filter_registry


"""
Parameter configurations
"""


@dataclass
class MeanFieldParameterApproximation:
    label: str = field(init=False, default="mean-field")


@dataclass
class MAFParameterApproximation:
    label: str = field(init=False, default="maf")
    nn_width: int = 32
    nn_depth: int = 2


@dataclass
class MultivariateNormalParameterApproximation:
    label: str = field(init=False, default="multivariate-normal")
    diag_jitter: float = 1e-6


ParameterApproximationLabels = typing.Literal[
    "mean-field", "maf", "multivariate-normal"
]
ParameterApproximation = (
    MeanFieldParameterApproximation
    | MAFParameterApproximation
    | MultivariateNormalParameterApproximation
)

parameter_approximation_registry: dict[
    ParameterApproximationLabels, type[ParameterApproximation]
] = {
    "mean-field": MeanFieldParameterApproximation,
    "maf": MAFParameterApproximation,
    "multivariate-normal": MultivariateNormalParameterApproximation,
}


def get_interval_spline():
    return transformations.Chain(
        (
            transformations.Sigmoid(lower=-1.0, upper=1.0),
            transformations.ConstrainedRQS(num_bins=5, lower=-1.0, upper=1.0),
        )
    )


configured_bijections: dict[str, typing.Callable[[], transformations.Bijector]] = {
    "interval_spline": get_interval_spline,
    "sigmoid": partial(transformations.Sigmoid, lower=-1. + 1e-6, upper=1. - 1e-6),
    "softplus": transformations.Softplus,
}

@dataclass
class DefaultTransform:
    pass

parameter_transform_registry = {
    "default": DefaultTransform,
}

BijectionConfiguration = DefaultTransform

def _build_parameter_approximation[
    InferenceParameterT: seqjtyping.Parameters,
](
    target_struct_cls: type[InferenceParameterT],
    approximation: ParameterApproximation,
    *,
    key: jaxtyping.PRNGKeyArray,
) -> base.UnconditionalVariationalApproximation[InferenceParameterT]:
    field_bijections: dict[str, transformations.Bijector] = {}

    constraint_factory = partial(
        transformations.FieldwiseBijector,
        field_bijections=field_bijections,
    )

    base_factory: typing.Callable[..., base.UnconditionalVariationalApproximation]
    if isinstance(approximation, MeanFieldParameterApproximation):
        base_factory = base.MeanField
    elif isinstance(approximation, MultivariateNormalParameterApproximation):
        base_factory = partial(
            base.MultivariateNormal,
            diag_jitter=approximation.diag_jitter,
        )
    elif isinstance(approximation, MAFParameterApproximation):
        base_factory =partial(
            maf.MaskedAutoregressiveFlow,
            key=key,
            nn_width=approximation.nn_width,
            nn_depth=approximation.nn_depth,
        ) 
    else:
        raise ValueError(f"Unsupported parameter approximation: {approximation}")

    return transformed.transform_approximation(
        target_struct_class=target_struct_cls,
        base=base_factory,
        constraint=constraint_factory,
    )


"""
Latent configurations
"""


@dataclass
class AutoregressiveLatentApproximation:
    label: str = field(init=False, default="autoregressive")
    nn_width: int = 32
    nn_depth: int = 2
    lag_order: int = 1


@dataclass
class MAFLatentApproximation:
    label: str = field(init=False, default="masked-autoregressive-flow")
    nn_width: int = 20
    nn_depth: int = 2
    base_loc: float = 0.0
    base_scale: float = 1.0
    flow_layers: int = 1

@dataclass
class ConvNFLatentApproximation:
    label: str = field(init=False, default="conv-flow")
    nn_width: int = 64
    nn_depth: int = 2
    radius: int = 2
    flow_layers: int = 2
    add_ar_layer: bool = False
    add_conditional_affine: bool = False

@dataclass
class StructuredPrecisionLatentApproximation:
    label: str = field(init=False, default="structured")
    temporal_structure: str = field(init=False, default="bidiagonal")
    within_time_structure: str = field(init=False, default="full")
    nn_width: int = 32
    nn_depth: int = 2

@dataclass
class MeanFieldLatentApproximation:
    label: str = field(init=False, default="mean-field")
    temporal_structure: str = field(init=False, default="mean_field")
    within_time_structure: str = field(init=False, default="full")
    nn_width: int = 32
    nn_depth: int = 2

LatentApproximation = (
    AutoregressiveLatentApproximation 
    | MAFLatentApproximation
    | StructuredPrecisionLatentApproximation
    | MeanFieldLatentApproximation
    | ConvNFLatentApproximation
)
LatentApproximationLabels = typing.Literal[
    "autoregressive", "masked-autoregressive-flow", "structured", "conv-flow"
]
latent_approximation_registry: dict[LatentApproximationLabels, type[LatentApproximation]] = {
    "autoregressive": AutoregressiveLatentApproximation,
    "masked-autoregressive-flow": MAFLatentApproximation,
    "structured": StructuredPrecisionLatentApproximation,
    "mean-field": MeanFieldLatentApproximation,
    "conv-flow": ConvNFLatentApproximation,
}


def build_latent_approximation(
    latent_config: LatentApproximation, 
    sample_length: int, 
    target_model: model_interface.SequentialModelProtocol,
    key: jaxtyping.PRNGKeyArray,
    latent_context_dims: LatentContextDims,
):
    target_latent_class = target_model.latent_cls

    if isinstance(latent_config, AutoregressiveLatentApproximation):
        if target_latent_class.flat_dim == 1:
            latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
                target_latent_class,
                sample_length=sample_length,
                latent_context_dims=latent_context_dims,
                lag_order=latent_config.lag_order,
                nn_width=latent_config.nn_width,
                nn_depth=latent_config.nn_depth,
                key=key,
            )
        else:
            latent_approximation = autoregressive.AmortizedMultivariateAutoregressor(
                target_latent_class,
                sample_length=sample_length,
                latent_context_dims=latent_context_dims,
                lag_order=latent_config.lag_order,
                nn_width=latent_config.nn_width,
                nn_depth=latent_config.nn_depth,
                key=key,
            )

    elif isinstance(latent_config, MAFLatentApproximation):
        latent_approximation = maf.AmortizedMAF(
            target_latent_class,
            sample_length=sample_length,
            latent_context_dims=latent_context_dims,
            key=key,
            nn_width=latent_config.nn_width,
            nn_depth=latent_config.nn_depth,
            flow_layers=latent_config.flow_layers,
            base_loc=latent_config.base_loc,
            base_scale=latent_config.base_scale,
        )

    elif (
        isinstance(latent_config, StructuredPrecisionLatentApproximation)
        or isinstance(latent_config, MeanFieldLatentApproximation)
    ):
        latent_approximation = structured.StructuredPrecisionGaussian(
            target_latent_class,
            sample_length=sample_length,
            latent_context_dims=latent_context_dims,
            temporal_structure=latent_config.temporal_structure,
            within_time_structure=latent_config.within_time_structure,
            hidden_dim=latent_config.nn_width,
            depth=latent_config.nn_depth,
            key=key,
        )

    elif isinstance(latent_config, ConvNFLatentApproximation):
        latent_approximation = conv_nf.AmortizedConvCoupling(
            target_latent_class,
            sample_length=sample_length,
            latent_context_dims=latent_context_dims,
            nn_width=latent_config.nn_width,
            nn_depth=latent_config.nn_depth,
            key=key,
            radius=latent_config.radius,
            flow_layers=latent_config.flow_layers,
            add_ar_layer=latent_config.add_ar_layer,
            add_conditional_affine=latent_config.add_conditional_affine,
        )

    else:
        raise ValueError(
            f"Unknown latent approximation configuration: {latent_config!r}"
        )
    
    return latent_approximation


"""
Approximations
"""


@dataclass
class FullVIConfig(VISampleConfig):
    optimization: optimization_registry.OptConfig
    embedder: EmbedderConfig
    samples_per_context: int
    num_sequence_minibatch: int = 1
    parameter_approximation: ParameterApproximation = field(
        default_factory=MeanFieldParameterApproximation
    )
    latent_approximation: LatentApproximation = field(
        default_factory=AutoregressiveLatentApproximation
    )
    pre_training_optimization: None | optimization_registry.OptConfig = None
    prior_training_optimization: None | optimization_registry.OptConfig = None
    sync_interval_s: None | int = None

    def training_sampling_kwargs(self, *, loss_label: str) -> VISamplingKwargs:
        return {
            "context_samples": 1,
            "samples_per_context": self.samples_per_context,
            "num_sequence_minibatch": self.num_sequence_minibatch,
        }

    def evaluation_sampling_kwargs(self, *, test_samples: int) -> VISamplingKwargs:
        return {
            "context_samples": 1,
            "samples_per_context": max(1, int(test_samples)),
            "num_sequence_minibatch": self.num_sequence_minibatch,
        }

@dataclass
class HybridVIConfig(VISampleConfig):
    optimization: optimization_registry.OptConfig
    particle_filter_config: particle_filter_registry.BootstrapFilterConfig
    samples_per_context: int
    buffer_length: int
    batch_length: int
    num_sequence_minibatch: int = 1
    parameter_approximation: ParameterApproximation = field(
        default_factory=MeanFieldParameterApproximation
    )
    prior_training_optimization: None | optimization_registry.OptConfig = None
    sync_interval_s: None | int = None

    def training_sampling_kwargs(self, *, loss_label: str) -> VISamplingKwargs:
        return {
            "context_samples": 1,
            "samples_per_context": self.samples_per_context,
            "num_sequence_minibatch": self.num_sequence_minibatch,
        }

    def evaluation_sampling_kwargs(self, *, test_samples: int) -> VISamplingKwargs:
        return {
            "context_samples": 1,
            "samples_per_context": max(1, int(test_samples)),
            "num_sequence_minibatch": self.num_sequence_minibatch,
        }
    
@dataclass
class BufferedVIConfig(VISampleConfig):
    optimization: optimization_registry.OptConfig
    buffer_length: int
    batch_length: int
    num_context_per_sequence: int
    samples_per_context: int
    embedder: EmbedderConfig
    num_sequence_minibatch: int = 1
    pre_training_optimization: None | optimization_registry.OptConfig = None
    parameter_approximation: ParameterApproximation = field(
        default_factory=MeanFieldParameterApproximation
    )
    latent_approximation: LatentApproximation = field(
        default_factory=AutoregressiveLatentApproximation
    )
    prior_training_optimization: None | optimization_registry.OptConfig = None
    loss_style: str = "standard"
    unroll: int = 1
    compiled_steps: int = 1
    sync_interval_s: None | int = None

    def training_sampling_kwargs(self, *, loss_label: str) -> VISamplingKwargs:
        return {
            "context_samples": self.num_context_per_sequence,
            "samples_per_context": self.samples_per_context,
            "num_sequence_minibatch": self.num_sequence_minibatch,
        }

    def evaluation_sampling_kwargs(self, *, test_samples: int) -> VISamplingKwargs:
        context_samples = max(1, min(self.num_context_per_sequence, int(test_samples)))
        samples_per_context = max(1, int(test_samples) // context_samples)
        return {
            "context_samples": context_samples,
            "samples_per_context": samples_per_context,
            "num_sequence_minibatch": self.num_sequence_minibatch,
        }

@eqx.nn.make_with_state
def build_approximation(
    config: FullVIConfig | BufferedVIConfig | HybridVIConfig,
    sequence_length: int,
    target_posterior: model_interface.BayesianSequentialModelProtocol,
    key: jaxtyping.PRNGKeyArray,
) -> tuple[typing.Any, eqx.nn.State]:
    parameter_key, approximation_key, embedding_key = jrandom.split(key, 3)

    target_param_class = target_posterior.parameterization.inference_parameter_cls

    parameter_approximation = _build_parameter_approximation(
        target_param_class,
    config.parameter_approximation,
        key=parameter_key,
    )

    if isinstance(config, FullVIConfig):
        embed = build_embedder(
            config.embedder,
            target_posterior.target,
            target_posterior.parameterization.inference_parameter_cls,
            sample_length=sequence_length,
            sequence_length=sequence_length,
            embedding_key=embedding_key,
        )
    elif isinstance(config, BufferedVIConfig):
        embed = build_embedder(
            config.embedder,
            target_posterior.target,
            target_posterior.parameterization.inference_parameter_cls,
            sequence_length=sequence_length,
            sample_length=config.batch_length + 2 * config.buffer_length,
            embedding_key=embedding_key,
        )
        

    approximation: typing.Any
    latent_approximation: (
        autoregressive.AmortizedUnivariateAutoregressor
        | maf.AmortizedMAF
        | structured.StructuredPrecisionGaussian
        | conv_nf.AmortizedConvCoupling
    )
    if isinstance(config, FullVIConfig):
        latent_approximation = build_latent_approximation(
            config.latent_approximation,
            sequence_length,
            target_model=target_posterior.target,
            latent_context_dims=embed.latent_context_dims,
            key=approximation_key
        )

        approximation = base.FullVI(
            latent_approximation,
            parameter_approximation,
            embed,
            target_posterior,
        )

    elif isinstance(config, BufferedVIConfig):
        latent_approximation = build_latent_approximation(
            config.latent_approximation,
            config.buffer_length * 2 + config.batch_length,
            target_model=target_posterior.target,
            latent_context_dims=embed.latent_context_dims,
            key=approximation_key
        )
        
        if config.loss_style == "standard":
            approximation = base.BufferedSSMVI(
                latent_approximation,
                parameter_approximation,
                embed,
                target_posterior,
                batch_length=config.batch_length,
                buffer_length=config.buffer_length,
            )
        elif config.loss_style == "inner-iw":
            approximation = base.IWBufferedSSMVI(
                latent_approximation,
                parameter_approximation,
                embed,
                target_posterior,
                batch_length=config.batch_length,
                buffer_length=config.buffer_length,
            )

    elif isinstance(config, HybridVIConfig):
        approximation = hybrid.HybridSSMVI(
            parameter_approximation=parameter_approximation,
            target_posterior=target_posterior,
            particle_filter=particle_filter_registry.build_filter(
                target_posterior,
                config.particle_filter_config,
            ),
            batch_length=config.batch_length,
            buffer_length=config.buffer_length,

        )

    return approximation
