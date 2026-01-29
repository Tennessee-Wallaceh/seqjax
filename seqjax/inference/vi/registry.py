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
from seqjax.inference import embedder
from seqjax.inference.vi import autoregressive
from seqjax.model.base import BayesianSequentialModel
from seqjax.model.registry import default_parameter_transforms

"""
Embedding configurations
"""
EmbedderName = typing.Literal["short-window", "long-window", "bi-rnn"]


@dataclass
class ShortContextEmbedder:
    label: EmbedderName = field(init=False, default="short-window")
    prev_window: int = 2
    post_window: int = 2
    
@dataclass
class LongContextEmbedder:
    label: EmbedderName = field(init=False, default="long-window")
    prev_window: int = 10
    post_window: int = 10


@dataclass
class BiRNNEmbedder:
    label: EmbedderName = field(init=False, default="bi-rnn")
    hidden_dim: int = 10


EmbedderConfig = type[ShortContextEmbedder] | type[LongContextEmbedder] | type[BiRNNEmbedder]

embedder_registry: dict[EmbedderName, EmbedderConfig] = {
    "short-window": ShortContextEmbedder,
    "long-window": LongContextEmbedder,
    "bi-rnn": BiRNNEmbedder,
}


def _build_embedder(
    embedder_config: EmbedderConfig,
    target_dim: int,
    sample_length: int,
    embedding_key: jaxtyping.PRNGKeyArray,
) -> embedder.Embedder:
    embed: embedder.Embedder
    if isinstance(embedder_config, ShortContextEmbedder):
        embed = embedder.WindowEmbedder(
            sample_length=sample_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
            y_dimension=target_dim,
        )
    elif isinstance(embedder_config, LongContextEmbedder):
        embed = embedder.WindowEmbedder(
            sample_length=sample_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
            y_dimension=target_dim,
        )
    elif isinstance(embedder_config, BiRNNEmbedder):
        embed = embedder.RNNEmbedder(
            embedder_config.hidden_dim,
            target_dim,
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
class MaskedAutoregressiveParameterApproximation:
    label: str = field(init=False, default="maf")
    nn_width: int = 32
    nn_depth: int = 2


ParameterApproximationLabels = typing.Literal["mean-field", "maf"]
ParameterApproximation = (
    MeanFieldParameterApproximation | MaskedAutoregressiveParameterApproximation
)

parameter_approximation_registry: dict[
    ParameterApproximationLabels, type[ParameterApproximation]
] = {
    "mean-field": MeanFieldParameterApproximation,
    "maf": MaskedAutoregressiveParameterApproximation,
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
    "sigmoid": partial(transformations.Sigmoid, lower=-1.0, upper=1.0),
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
    bijection_configuration = DefaultTransform
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
    elif isinstance(approximation, MaskedAutoregressiveParameterApproximation):
        base_factory = base.MaskedAutoregressiveFlowFactory[ParametersT](
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
    nn_width: int = 20
    nn_depth: int = 2
    lag_order: int = 1


@dataclass
class MaskedAutoregressiveFlowLatentApproximation:
    label: str = field(init=False, default="masked-autoregressive-flow")
    nn_width: int = 20
    nn_depth: int = 2
    conditioner_width: int = 32
    conditioner_depth: int = 2
    conditioner_out_dim: int = 32
    base_loc: float = 0.0
    base_scale: float = 1.0


LatentApproximation = (
    AutoregressiveLatentApproximation | MaskedAutoregressiveFlowLatentApproximation
)
LatentApproximationLabels = typing.Literal[
    "autoregressive", "masked-autoregressive-flow"
]
latent_approximation_registry: dict[LatentApproximationLabels, LatentApproximation] = {
    "autoregressive": AutoregressiveLatentApproximation(),
    "masked-autoregressive-flow": MaskedAutoregressiveFlowLatentApproximation(),
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


def build_approximation(
    config: FullVIConfig | BufferedVIConfig,
    sequence_length: int,
    target_posterior: BayesianSequentialModel,
    key: jaxtyping.PRNGKeyArray,
) -> base.SSMVariationalApproximation:
    parameter_key, approximation_key, embedding_key = jrandom.split(key, 3)

    target_observation_class = target_posterior.target.observation_cls
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
            target_observation_class.flat_dim,
            sequence_length,
            embedding_key,
        )
    elif isinstance(config, BufferedVIConfig):
        embed = _build_embedder(
            config.embedder,
            target_observation_class.flat_dim,
            config.batch_length + 2 * config.buffer_length,
            embedding_key,
        )

    approximation: base.SSMVariationalApproximation
    latent_approximation: (
        autoregressive.AmortizedUnivariateAutoregressor
        | base.AmortizedMaskedAutoregressiveFlow
    )
    if isinstance(config, FullVIConfig):
        latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
            target_latent_class,
            buffer_length=0,
            batch_length=sequence_length,
            context_dim=embed.context_dimension + 1,  # just location
            parameter_dim=target_param_class.flat_dim,
            condition_dim=target_posterior.target.condition_cls.flat_dim,
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
                buffer_length=config.buffer_length,
                batch_length=config.batch_length,
                context_dim=embed.context_dimension,
                parameter_dim=target_param_class.flat_dim,
                condition_dim=target_posterior.target.condition_cls.flat_dim,
                lag_order=latent_config.lag_order,
                nn_width=latent_config.nn_width,
                nn_depth=latent_config.nn_depth,
                key=approximation_key,
            )

            # latent_approximation = (
            #     autoregressive.AmortizedInnovationUnivariateAutoregressor(
            #         target_posterior,
            #         buffer_length=config.buffer_length,
            #         batch_length=config.batch_length,
            #         context_dim=embed.context_dimension,
            #         parameter_dim=target_param_class.flat_dim,
            #         condition_dim=target_posterior.target.condition_cls.flat_dim,
            #         lag_order=latent_config.lag_order,
            #         nn_width=latent_config.nn_width,
            #         nn_depth=latent_config.nn_depth,
            #         key=approximation_key,
            #     )
            # )

        elif isinstance(latent_config, MaskedAutoregressiveFlowLatentApproximation):
            latent_approximation = base.AmortizedMaskedAutoregressiveFlow(
                target_latent_class,
                buffer_length=config.buffer_length,
                batch_length=config.batch_length,
                context_dim=embed.context_dimension,
                parameter_dim=target_param_class.flat_dim,
                key=approximation_key,
                nn_width=latent_config.nn_width,
                nn_depth=latent_config.nn_depth,
                conditioner_width=latent_config.conditioner_width,
                conditioner_depth=latent_config.conditioner_depth,
                conditioner_out_dim=latent_config.conditioner_out_dim,
                base_loc=latent_config.base_loc,
                base_scale=latent_config.base_scale,
            )
        else:
            raise ValueError(
                f"Unknown latent approximation configuration: {latent_config!r}"
            )
        approximation = base.BufferedSSMVI(
            latent_approximation,
            parameter_approximation,
            embed,
            control_variate=config.control_variate,
        )
    return approximation
