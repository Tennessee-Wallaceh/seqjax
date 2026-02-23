import typing
from dataclasses import dataclass, field
from functools import partial

import jax.random as jrandom
import jaxtyping

import seqjax.model.typing as seqjtyping
from seqjax.inference.optimization import registry as optimization_registry
from seqjax.inference.vi import transformations
from seqjax.inference.vi import transformed
from seqjax.inference.vi import base
from seqjax.inference.vi import embedder
from seqjax.inference.vi import maf
from seqjax.inference.vi import autoregressive
from seqjax.inference.vi import structured
from seqjax.model.base import BayesianSequentialModel
from seqjax.model.registry import default_parameter_transforms

"""
Embedding configurations
"""
EmbedderName = typing.Literal[
    "short-window", "long-window", "bi-rnn", "passthrough", "conv1d", "transformer"
]


@dataclass
class PassthroughEmbedder:
    label: EmbedderName = field(init=False, default="passthrough")
    prev_window: int = field(init=False, default=0)
    post_window: int = field(init=False, default=0)

@dataclass
class ShortContextEmbedder:
    label: EmbedderName = field(init=False, default="short-window")
    prev_window: int = field(init=False, default=2)
    post_window: int = field(init=False, default=2)
    
@dataclass
class LongContextEmbedder:
    label: EmbedderName = field(init=False, default="long-window")
    prev_window: int = 10
    post_window: int = 10

@dataclass
class Conv1DEmbedderConfig:
    label: EmbedderName = field(init=False, default="conv1d")
    hidden_dim: int = 2
    kernel_size: int = 3
    depth: int = 2
    pool_dim: None | int = None
    pool_kind: str = "avg"

@dataclass
class BiRNNEmbedder:
    label: EmbedderName = field(init=False, default="bi-rnn")
    hidden_dim: int = 10


@dataclass
class TransformerEmbedderConfig:
    label: EmbedderName = field(init=False, default="transformer")
    hidden_dim: int = 32
    depth: int = 2
    num_heads: int = 2
    mlp_multiplier: int = 4
    pool_dim: None | int = None


EmbedderConfig = (
    ShortContextEmbedder 
    | LongContextEmbedder 
    | BiRNNEmbedder 
    | PassthroughEmbedder
    | Conv1DEmbedderConfig
    | TransformerEmbedderConfig
)

embedder_registry: dict[EmbedderName, type[EmbedderConfig]] = {
    "short-window": ShortContextEmbedder,
    "long-window": LongContextEmbedder,
    "bi-rnn": BiRNNEmbedder,
    "passthrough": PassthroughEmbedder,
    "conv1d": Conv1DEmbedderConfig,
    "transformer": TransformerEmbedderConfig,
}

def _build_embedder(
    embedder_config: EmbedderConfig,
    target_posterior: BayesianSequentialModel,
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
        )
    elif isinstance(embedder_config, PassthroughEmbedder):
        embed = embedder.WindowEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
        )
    elif isinstance(embedder_config, LongContextEmbedder):
        embed = embedder.WindowEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
        )
    elif isinstance(embedder_config, Conv1DEmbedderConfig):
        embed = embedder.Conv1DEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            hidden=embedder_config.hidden_dim,
            kernel_size=embedder_config.kernel_size,
            depth=embedder_config.depth,
            key=embedding_key,
            pool_dim=embedder_config.pool_dim,
            pool_kind=embedder_config.pool_kind,
        )
    elif isinstance(embedder_config, BiRNNEmbedder):
        embed = embedder.RNNEmbedder(
            target_posterior=target_posterior,
            sample_length=sample_length,
            sequence_length=sequence_length,
            hidden=embedder_config.hidden_dim,
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
            key=embedding_key,
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder_config.label}")

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
    ParametersT: seqjtyping.Parameters,
](
    target_struct_cls: type[ParametersT],
    approximation: ParameterApproximation,
    *,
    key: jaxtyping.PRNGKeyArray,
) -> base.UnconditionalVariationalApproximation[ParametersT]:
    # TODO: allow other bijection configurations- I now think this should
    # work via a map of options on seqjtyping.Parameter classes.
    # Ie have a "default"/"spline" string switch
    # this is part of inference, so could be in an inference.reparametrization submodule
    bijection_configuration = DefaultTransform()
    # handle parameter constrainsts with specified constraint transforms
    if isinstance(bijection_configuration, DefaultTransform):
        field_transforms = default_parameter_transforms[
            target_struct_cls
        ]
    else:
        raise Exception("Unknown bijection configuration")

    field_bijections: dict[str, transformations.Bijector] = {}
    for parameter_field, bijection in field_transforms.items():
        field_bijections[parameter_field] = configured_bijections[bijection]()


    constraint_factory = partial(
        transformations.FieldwiseBijector,
        field_bijections=field_bijections,
    )

    if isinstance(approximation, MeanFieldParameterApproximation):
        base_factory: base.VariationalApproximationFactory[ParametersT, None] = (
            base.MeanField
        )
    elif isinstance(approximation, MultivariateNormalParameterApproximation):
        base_factory = partial(
            base.MultivariateNormal,
            diag_jitter=approximation.diag_jitter,
        )
    elif isinstance(approximation, MAFParameterApproximation):
        base_factory =partial(
            maf.MaskedAutoregressiveFlow,
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
    nn_width: int = 20
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
class StructuredPrecisionLatentApproximation:
    label: str = field(init=False, default="structured")
    nn_width: int = 32
    nn_depth: int = 2

LatentApproximation = (
    AutoregressiveLatentApproximation 
    | MAFLatentApproximation
    | StructuredPrecisionLatentApproximation
)
LatentApproximationLabels = typing.Literal[
    "autoregressive", "masked-autoregressive-flow", "structured"
]
latent_approximation_registry: dict[LatentApproximationLabels, type[LatentApproximation]] = {
    "autoregressive": AutoregressiveLatentApproximation,
    "masked-autoregressive-flow": MAFLatentApproximation,
    "structured": StructuredPrecisionLatentApproximation,
}

"""
Approximations
"""


@dataclass
class FullVIConfig:
    optimization: optimization_registry.OptConfig
    parameter_field_bijections: dict[str, str | transformations.Bijector]
    embedder: EmbedderConfig
    observations_per_step: int
    samples_per_context: int
    parameter_approximation: ParameterApproximation


@dataclass
class BufferedVIConfig:
    optimization: optimization_registry.OptConfig
    buffer_length: int
    batch_length: int
    observations_per_step: int
    samples_per_context: int
    embedder: EmbedderConfig
    control_variate: bool = False
    pre_training_optimization: None | optimization_registry.OptConfig = None
    parameter_approximation: ParameterApproximation = field(
        default_factory=MeanFieldParameterApproximation
    )
    latent_approximation: LatentApproximation = field(
        default_factory=AutoregressiveLatentApproximation
    )
    prior_training_optimization: None | optimization_registry.OptConfig = None

def build_approximation(
    config: FullVIConfig | BufferedVIConfig,
    sequence_length: int,
    target_posterior: BayesianSequentialModel,
    key: jaxtyping.PRNGKeyArray,
) -> base.SSMVariationalApproximation:
    parameter_key, approximation_key, embedding_key = jrandom.split(key, 3)

    target_param_class = target_posterior.inference_parameter_cls
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

    approximation: base.SSMVariationalApproximation
    latent_approximation: (
        autoregressive.AmortizedUnivariateAutoregressor
        | maf.AmortizedMAF
        | structured.StructuredPrecisionGaussian
    )
    if isinstance(config, FullVIConfig):
        latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
            target_latent_class,
            sample_length=sequence_length,
            embedder=embed,
            lag_order=1,
            nn_width=20,
            nn_depth=2,
            key=approximation_key,
        )

        approximation = base.FullAutoregressiveVI(
            latent_approximation,
            parameter_approximation,
            embed,
        )

    elif isinstance(config, BufferedVIConfig):
        latent_config = config.latent_approximation

        if isinstance(latent_config, AutoregressiveLatentApproximation):
            latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
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

        else:
            raise ValueError(
                f"Unknown latent approximation configuration: {latent_config!r}"
            )
        approximation = base.BufferedSSMVI(
            latent_approximation,
            parameter_approximation,
            embed,
            batch_length=config.batch_length,
            buffer_length=config.buffer_length,
            control_variate=config.control_variate,
        )
    return approximation
