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
from seqjax.inference.vi import embedder
from seqjax.inference.vi import aggregation
from seqjax.inference.vi import maf
from seqjax.inference.vi import conv_nf
from seqjax.inference.vi import autoregressive
from seqjax.inference.vi import structured
from seqjax.inference.vi.sampling import VISampleConfig, VISamplingKwargs
from seqjax.model.interface import BayesianSequentialModelProtocol
from seqjax.inference.particlefilter import registry as particle_filter_registry

"""
Embedding configurations
"""
PositionMode = typing.Literal["sample", "sequence"]


EmbedderName = typing.Literal[
    "short-window", "long-window", "bi-rnn", "passthrough", "conv1d", "transformer", "positional"
]


@dataclass
class PassthroughEmbedder:
    label: EmbedderName = field(init=False, default="passthrough")
    prev_window: int = field(init=False, default=0)
    post_window: int = field(init=False, default=0)
    position_mode: None | PositionMode = None
    n_pos_embedding: int = 8

@dataclass
class ShortContextEmbedder:
    label: EmbedderName = field(init=False, default="short-window")
    prev_window: int = field(init=False, default=2)
    post_window: int = field(init=False, default=2)
    position_mode: None | PositionMode = None
    n_pos_embedding: int = 8
    
@dataclass
class LongContextEmbedder:
    label: EmbedderName = field(init=False, default="long-window")
    prev_window: int = 10
    post_window: int = 10
    position_mode: None | PositionMode = None
    n_pos_embedding: int = 1

@dataclass
class Conv1DEmbedderConfig:
    label: EmbedderName = field(init=False, default="conv1d")
    hidden_dim: int = 2
    kernel_size: int = 3
    depth: int = 2
    pool_dim: None | int = None
    pool_kind: str = "avg"
    position_mode: None | PositionMode = None
    n_pos_embedding: int = 1
    embed_norm_kind: None | str = None
    param_norm: bool = False

@dataclass
class BiRNNEmbedder:
    label: EmbedderName = field(init=False, default="bi-rnn")
    hidden_dim: int = 10
    aggregation_kind: aggregation.AggregationKind = "observation-flatten"
    position_mode: None | PositionMode = None
    n_pos_embedding: int = 1


@dataclass
class TransformerEmbedderConfig:
    label: EmbedderName = field(init=False, default="transformer")
    hidden_dim: int = 32
    depth: int = 2
    num_heads: int = 2
    mlp_multiplier: int = 4
    pool_dim: None | int = None
    position_mode: None | PositionMode = None
    n_pos_embedding: int = 8


@dataclass
class PositionalEmbedderConfig:
    label: EmbedderName = field(init=False, default="positional")
    n_pos_embedding: int = 8
    position_mode: PositionMode = "sample"


EmbedderConfig = (
    ShortContextEmbedder 
    | LongContextEmbedder 
    | BiRNNEmbedder 
    | PassthroughEmbedder
    | Conv1DEmbedderConfig
    | TransformerEmbedderConfig
    | PositionalEmbedderConfig
)

embedder_registry: dict[EmbedderName, type[EmbedderConfig]] = {
    "short-window": ShortContextEmbedder,
    "long-window": LongContextEmbedder,
    "bi-rnn": BiRNNEmbedder,
    "passthrough": PassthroughEmbedder,
    "conv1d": Conv1DEmbedderConfig,
    "transformer": TransformerEmbedderConfig,
    "positional": PositionalEmbedderConfig,
}

def _build_embedder(
    embedder_config: EmbedderConfig,
    target_posterior: BayesianSequentialModelProtocol,
    sequence_length: int,
    sample_length: int,
    embedding_key: jaxtyping.PRNGKeyArray,
) -> embedder.Embedder:
    embed: embedder.Embedder
    if isinstance(embedder_config, ShortContextEmbedder):
        embed = embedder.WindowEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
        )
    elif isinstance(embedder_config, PassthroughEmbedder):
        embed = embedder.WindowEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
        )
    elif isinstance(embedder_config, LongContextEmbedder):
        embed = embedder.WindowEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
        )
    elif isinstance(embedder_config, Conv1DEmbedderConfig):
        embed = embedder.Conv1DEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            hidden=embedder_config.hidden_dim,
            kernel_size=embedder_config.kernel_size,
            depth=embedder_config.depth,
            embed_norm_kind=embedder_config.embed_norm_kind,
            key=embedding_key,
            pool_dim=embedder_config.pool_dim,
            pool_kind=embedder_config.pool_kind,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
            use_param_norm=embedder_config.param_norm,
        )
    elif isinstance(embedder_config, BiRNNEmbedder):
        embed = embedder.RNNEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            hidden=embedder_config.hidden_dim,
            aggregation_kind=embedder_config.aggregation_kind,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
            key=embedding_key,
        )
    elif isinstance(embedder_config, TransformerEmbedderConfig):
        embed = embedder.TransformerEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            hidden=embedder_config.hidden_dim,
            depth=embedder_config.depth,
            num_heads=embedder_config.num_heads,
            mlp_multiplier=embedder_config.mlp_multiplier,
            pool_dim=embedder_config.pool_dim,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
            key=embedding_key,
        )
    elif isinstance(embedder_config, PositionalEmbedderConfig):
        embed = embedder.PositionalEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            n_pos_embedding=embedder_config.n_pos_embedding,
            position_mode=embedder_config.position_mode,
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder_config.label}")

   
    print(
        f"observation_context_dim: {embed.observation_context_dim} \n",
        f"condition_context_dim: {embed.condition_context_dim} \n",
        f"parameter_context_dim: {embed.parameter_context_dim} \n",
        f"embedded_context_dim: {embed.embedded_context_dim} \n",
        f"sequence_embedded_context_dim: {embed.sequence_embedded_context_dim} \n",
    )
    return embed


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
    nn_width: int = 32
    nn_depth: int = 2
    kernel_size: int = 5
    flow_layers: int = 2

@dataclass
class StructuredPrecisionLatentApproximation:
    label: str = field(init=False, default="structured")
    nn_width: int = 32
    nn_depth: int = 2

LatentApproximation = (
    AutoregressiveLatentApproximation 
    | MAFLatentApproximation
    | StructuredPrecisionLatentApproximation
    | ConvNFLatentApproximation
)
LatentApproximationLabels = typing.Literal[
    "autoregressive", "masked-autoregressive-flow", "structured", "conv-flow"
]
latent_approximation_registry: dict[LatentApproximationLabels, type[LatentApproximation]] = {
    "autoregressive": AutoregressiveLatentApproximation,
    "masked-autoregressive-flow": MAFLatentApproximation,
    "structured": StructuredPrecisionLatentApproximation,
    "conv-flow": ConvNFLatentApproximation,
}

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
    target_posterior: BayesianSequentialModelProtocol,
    key: jaxtyping.PRNGKeyArray,
) -> tuple[typing.Any, eqx.nn.State]:
    parameter_key, approximation_key, embedding_key = jrandom.split(key, 3)

    target_param_class = target_posterior.parameterization.inference_parameter_cls
    target_latent_class = target_posterior.target.latent_cls

    parameter_approximation = _build_parameter_approximation(
        target_param_class,
    config.parameter_approximation,
        key=parameter_key,
    )

    if isinstance(config, FullVIConfig):
        embed = _build_embedder(
            config.embedder,
            target_posterior,
            sample_length=sequence_length,
            sequence_length=sequence_length,
            embedding_key=embedding_key,
        )
    elif isinstance(config, BufferedVIConfig):
        embed = _build_embedder(
            config.embedder,
            target_posterior,
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
        latent_config = config.latent_approximation

        if isinstance(latent_config, AutoregressiveLatentApproximation):
            if target_latent_class.flat_dim == 1:
                latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
                    target_latent_class,
                    sample_length=sequence_length,
                    embedder=embed,
                    lag_order=latent_config.lag_order,
                    nn_width=latent_config.nn_width,
                    nn_depth=latent_config.nn_depth,
                    key=approximation_key,
                )
            else:
                latent_approximation = autoregressive.AmortizedMultivariateAutoregressor(
                    target_latent_class,
                    sample_length=sequence_length,
                    embedder=embed,
                    lag_order=latent_config.lag_order,
                    nn_width=latent_config.nn_width,
                    nn_depth=latent_config.nn_depth,
                    key=approximation_key,
                )
        elif isinstance(latent_config, MAFLatentApproximation):
            latent_approximation = maf.AmortizedMAF(
                target_latent_class,
                sample_length=sequence_length,
                embedder=embed,
                key=approximation_key,
                nn_width=latent_config.nn_width,
                nn_depth=latent_config.nn_depth,
                flow_layers=latent_config.flow_layers,
                base_loc=latent_config.base_loc,
                base_scale=latent_config.base_scale,
            )
        elif isinstance(latent_config, StructuredPrecisionLatentApproximation):
            latent_approximation = structured.StructuredPrecisionGaussian(
                target_latent_class,
                sample_length=sequence_length,
                embedder=embed,
                hidden_dim=latent_config.nn_width,
                depth=latent_config.nn_depth,
                key=approximation_key,
            )
        else:
            raise ValueError(
                f"Unknown latent approximation configuration: {latent_config!r}"
            )

        approximation = base.FullVI(
            latent_approximation,
            parameter_approximation,
            embed,
            target_posterior,
        )

    elif isinstance(config, BufferedVIConfig):
        latent_config = config.latent_approximation

        if isinstance(latent_config, AutoregressiveLatentApproximation):
            if target_latent_class.flat_dim == 1:
                latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
                    target_latent_class,
                    sample_length=config.buffer_length * 2 + config.batch_length,
                    embedder=embed,
                    lag_order=latent_config.lag_order,
                    nn_width=latent_config.nn_width,
                    nn_depth=latent_config.nn_depth,
                    key=approximation_key,
                )
            else:
                latent_approximation = autoregressive.AmortizedMultivariateAutoregressor(
                    target_latent_class,
                    sample_length=config.buffer_length * 2 + config.batch_length,
                    embedder=embed,
                    lag_order=latent_config.lag_order,
                    nn_width=latent_config.nn_width,
                    nn_depth=latent_config.nn_depth,
                    key=approximation_key,
                )

        elif isinstance(latent_config, MAFLatentApproximation):
            latent_approximation = maf.AmortizedMAF(
                target_latent_class,
                sample_length=config.buffer_length * 2 + config.batch_length,
                embedder=embed,
                key=approximation_key,
                nn_width=latent_config.nn_width,
                nn_depth=latent_config.nn_depth,
                flow_layers=latent_config.flow_layers,
                base_loc=latent_config.base_loc,
                base_scale=latent_config.base_scale,
            )

        elif isinstance(latent_config, StructuredPrecisionLatentApproximation):
            latent_approximation = structured.StructuredPrecisionGaussian(
                target_latent_class,
                sample_length=config.buffer_length * 2 + config.batch_length,
                embedder=embed,
                hidden_dim=latent_config.nn_width,
                depth=latent_config.nn_depth,
                key=approximation_key,
            )

        elif isinstance(latent_config, ConvNFLatentApproximation):
            latent_approximation = conv_nf.AmortizedConvCoupling(
                target_latent_class,
                sample_length=config.buffer_length * 2 + config.batch_length,
                embedder=embed,
                nn_width=latent_config.nn_width,
                nn_depth=latent_config.nn_depth,
                key=approximation_key,
                kernel_size=latent_config.kernel_size,
                flow_layers=latent_config.flow_layers,
            )

        else:
            raise ValueError(
                f"Unknown latent approximation configuration: {latent_config!r}"
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
