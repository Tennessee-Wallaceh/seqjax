import typing
from seqjax.inference.vi import transformed
from seqjax.inference.vi import base
from seqjax.inference.vi import transformations
from seqjax.inference import embedder
from seqjax.inference.vi import autoregressive
from seqjax.inference.vi import train
import optax  # type: ignore
import jax
import jax.numpy as jnp
from seqjax.model.base import BayesianSequentialModel
import seqjax.model.typing as seqjtyping
from seqjax.inference.interface import inference_method

import jaxtyping
import jax.random as jrandom
from functools import partial
from dataclasses import dataclass, field


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


EmbedderConfig = ShortContextEmbedder | LongContextEmbedder | BiRNNEmbedder

embedder_registry: dict[EmbedderName, EmbedderConfig] = {
    "short-window": ShortContextEmbedder(),
    "long-window": LongContextEmbedder(),
    "bi-rnn": BiRNNEmbedder(),
}

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
    ParameterApproximationLabels, ParameterApproximation
] = {
    "mean-field": MeanFieldParameterApproximation(),
    "maf": MaskedAutoregressiveParameterApproximation(),
}

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
Optimization configurations
"""


@dataclass
class CosineOpt:
    label: str = field(init=False, default="cosine-sched")
    warmup_steps: int = 0
    decay_steps: int = 5000
    peak_lr: float = 1e-2
    end_lr: float = 1e-5
    total_steps: int = 10_000
    time_limit_s: int | None = None

    def __repr__(self) -> str:
        return f"{self.label}({self.peak_lr:.0e},{self.end_lr:.0e},{self.warmup_steps},{self.decay_steps})"

    @classmethod
    def from_dict(cls, config_dict: dict[str, typing.Any]) -> "CosineOpt":
        return cls(
            warmup_steps=config_dict.get("warmup_steps", 0),
            decay_steps=config_dict.get("decay_steps", 5000),
            peak_lr=config_dict.get("peak_lr", 1e-2),
            end_lr=config_dict.get("end_lr", 1e-5),
            total_steps=config_dict.get("total_steps", 10_000),
            time_limit_s=config_dict.get("time_limit_s", None),
        )


@dataclass
class AdamOpt:
    label: str = field(init=False, default="adam-plain")
    lr: float = 1e-3
    total_steps: int = 500_000
    time_limit_s: int | None = 60 * 60 * 2  # default to 2 hour limit

    def __repr__(self) -> str:
        return f"{self.label}({self.lr:.0e})"

    @classmethod
    def from_dict(cls, config_dict: dict[str, typing.Any]) -> "AdamOpt":
        return cls(
            lr=config_dict.get("lr", 1e-3),
            total_steps=config_dict.get("total_steps", 500_000),
            time_limit_s=config_dict.get("time_limit_s", None),
        )


OptConfigLabels = typing.Literal["cosine-sched", "adam-plain"]
OptConfig = CosineOpt | AdamOpt
opt_config_registry: dict[OptConfigLabels, OptConfig] = {
    "cosine-sched": CosineOpt(),
    "adam-plain": AdamOpt(),
}


@dataclass
class FullVIConfig:
    optimization: OptConfig = field(default_factory=AdamOpt)
    parameter_field_bijections: dict[str, str | transformations.Bijector] = field(
        default_factory=dict
    )
    embedder: EmbedderConfig = field(default_factory=ShortContextEmbedder)
    observations_per_step: int = 10
    samples_per_context: int = 5
    parameter_approximation: ParameterApproximation = field(
        default_factory=MeanFieldParameterApproximation
    )
    num_tracker_steps: int = 100

    @classmethod
    def from_dict(cls, config_dict: dict[str, typing.Any]) -> "FullVIConfig":
        if config_dict["embedder"]["label"] not in embedder_registry:
            raise ValueError(
                f"Unknown embedder type: {config_dict['embedder']}. "
                f"Available embedders: {list(embedder_registry.keys())}"
            )

        if (
            config_dict["parameter_approximation"]["label"]
            not in parameter_approximation_registry
        ):
            raise ValueError(
                f"Unknown parameter approximation type: {config_dict['parameter_approximation']}. "
                f"Available parameter approximations: {list(parameter_approximation_registry.keys())}"
            )

        if config_dict["optimization"]["label"] not in opt_config_registry:
            raise ValueError(
                f"Unknown optimization type: {config_dict['optimization']}. "
                f"Available optimizations: {list(opt_config_registry.keys())}"
            )

        optimization = opt_config_registry[
            config_dict["optimization"]["label"]
        ].from_dict(config_dict["optimization"])

        return cls(
            optimization=optimization,
            parameter_field_bijections=config_dict["parameter_field_bijections"],
            embedder=embedder_registry[config_dict["embedder"]["label"]],
            observations_per_step=config_dict["observations_per_step"],
            samples_per_context=config_dict["samples_per_context"],
            parameter_approximation=parameter_approximation_registry[
                config_dict["parameter_approximation"]["label"]
            ],
        )


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


def _build_parameter_approximation[
    ParametersT: seqjtyping.Parameters,
](
    target_struct_cls: type[ParametersT],
    approximation: ParameterApproximation,
    field_bijections: dict[str, transformations.Bijector],
    *,
    key: jaxtyping.PRNGKeyArray,
) -> base.UnconditionalVariationalApproximation[ParametersT]:
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


def _build_embedder(
    embedder_config: EmbedderConfig,
    target_dim: int,
    sequence_length: int,
    embedding_key: jaxtyping.PRNGKeyArray,
) -> embedder.Embedder:
    embed: embedder.Embedder
    if embedder_config.label == "short-window":
        embed = embedder.WindowEmbedder(
            sequence_length,
            embedder_config.prev_window,
            embedder_config.post_window,
            target_dim,
        )
    elif embedder_config.label == "long-window":
        embed = embedder.WindowEmbedder(
            sequence_length,
            embedder_config.prev_window,
            embedder_config.post_window,
            target_dim,
        )
    elif embedder_config.label == "bi-rnn":
        embed = embedder.RNNEmbedder(
            embedder_config.hidden_dim,
            target_dim,
            key=embedding_key,
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder_config.label}")

    return embed


@dataclass
class BufferedVIConfig:
    optimization: OptConfig = field(default_factory=AdamOpt)
    parameter_field_bijections: dict[str, str] = field(default_factory=dict)
    buffer_length: int = 15
    batch_length: int = 10
    observations_per_step: int = 10
    samples_per_context: int = 5
    control_variate: bool = False
    pre_training_steps: int = 0
    embedder: EmbedderConfig = field(default_factory=ShortContextEmbedder)
    parameter_approximation: ParameterApproximation = field(
        default_factory=MeanFieldParameterApproximation
    )
    latent_approximation: LatentApproximation = field(
        default_factory=AutoregressiveLatentApproximation
    )
    num_tracker_steps: int = 100

    @classmethod
    def from_dict(cls, config_dict: dict[str, typing.Any]) -> "BufferedVIConfig":
        if config_dict["embedder"]["label"] not in embedder_registry:
            raise ValueError(
                f"Unknown embedder type: {config_dict['embedder']}. "
                f"Available embedders: {list(embedder_registry.keys())}"
            )

        if (
            config_dict["parameter_approximation"]["label"]
            not in parameter_approximation_registry
        ):
            raise ValueError(
                f"Unknown parameter approximation type: {config_dict['parameter_approximation']}. "
                f"Available parameter approximations: {list(parameter_approximation_registry.keys())}"
            )

        if (
            config_dict["latent_approximation"]["label"]
            not in latent_approximation_registry
        ):
            raise ValueError(
                f"Unknown latent approximation type: {config_dict['latent_approximation']}. "
                f"Available latent approximations: {list(latent_approximation_registry.keys())}"
            )

        if config_dict["optimization"]["label"] not in opt_config_registry:
            raise ValueError(
                f"Unknown optimization type: {config_dict['optimization']}. "
                f"Available optimizations: {list(opt_config_registry.keys())}"
            )

        optimization = opt_config_registry[
            config_dict["optimization"]["label"]
        ].from_dict(config_dict["optimization"])

        return cls(
            optimization=optimization,
            parameter_field_bijections=config_dict["parameter_field_bijections"],
            buffer_length=config_dict["buffer_length"],
            batch_length=config_dict["batch_length"],
            observations_per_step=config_dict["observations_per_step"],
            samples_per_context=config_dict["samples_per_context"],
            control_variate=config_dict["control_variate"],
            pre_training_steps=config_dict["pre_training_steps"],
            embedder=embedder_registry[config_dict["embedder"]["label"]],
            parameter_approximation=parameter_approximation_registry[
                config_dict["parameter_approximation"]["label"]
            ],
            latent_approximation=latent_approximation_registry[
                config_dict["latent_approximation"]["label"]
            ],
            num_tracker_steps=config_dict.get("num_tracker_steps", 100),
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
    target_latent_class = target_posterior.target.particle_cls

    # handle parameter constrainsts with specified constraint transforms
    field_bijections = {}
    for parameter_field, bijection in config.parameter_field_bijections.items():
        if isinstance(bijection, str):
            field_bijections[parameter_field] = configured_bijections[bijection]()
        else:
            field_bijections[parameter_field] = bijection

    parameter_approximation = _build_parameter_approximation(
        target_param_class,
        config.parameter_approximation,
        field_bijections,
        key=parameter_key,
    )
    embed = _build_embedder(
        config.embedder,
        target_observation_class.flat_dim,
        sequence_length,
        embedding_key,
    )

    if isinstance(config, FullVIConfig):
        latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
            target_latent_class,
            buffer_length=0,
            batch_length=sequence_length,
            context_dim=embed.context_dimension + 1,  # just location
            parameter_dim=target_param_class.flat_dim,
            lag_order=1,
            nn_width=10,
            nn_depth=2,
            key=approximation_key,
        )

        approximation: base.SSMVariationalApproximation = base.FullAutoregressiveVI(
            latent_approximation,
            parameter_approximation,
            embed,
        )

    elif isinstance(config, BufferedVIConfig):
        latent_config = config.latent_approximation
        latent_approximation: base.AmortizedVariationalApproximation

        if isinstance(latent_config, AutoregressiveLatentApproximation):
            latent_approximation = autoregressive.AmortizedUnivariateAutoregressor(
                target_latent_class,
                buffer_length=config.buffer_length,
                batch_length=config.batch_length,
                context_dim=embed.context_dimension,
                parameter_dim=target_param_class.flat_dim,
                lag_order=latent_config.lag_order,
                nn_width=latent_config.nn_width,
                nn_depth=latent_config.nn_depth,
                key=approximation_key,
            )
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
        approximation: base.SSMVariationalApproximation = base.BufferedSSMVI(
            latent_approximation,
            parameter_approximation,
            embed,
            control_variate=config.control_variate,
        )
    return approximation


@inference_method
def run_full_path_vi[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        ParticleT,
        InitialParticleT,
        TransitionParticleHistoryT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    hyperparameters: HyperParametersT,
    key: jaxtyping.PRNGKeyArray,
    observation_path: ObservationT,
    condition_path: ConditionT | None = None,
    test_samples: int = 1000,
    config: FullVIConfig = FullVIConfig(),
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    sequence_length = observation_path.batch_shape[0]

    approximation = build_approximation(
        config,
        sequence_length,
        target_posterior,
        key,
    )

    if isinstance(config.optimization, AdamOpt):
        optim = optax.apply_if_finite(
            optax.adam(config.optimization.lr), max_consecutive_errors=100
        )
    elif isinstance(config.optimization, CosineOpt):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.optimization.peak_lr,
            peak_value=config.optimization.peak_lr,
            warmup_steps=config.optimization.warmup_steps,
            decay_steps=config.optimization.decay_steps,
            end_value=config.optimization.end_lr,
        )

        optim = optax.apply_if_finite(
            optax.adam(learning_rate=schedule), max_consecutive_errors=100
        )

    fitted_approximation, opt_state = train.train(
        model=approximation,
        observations=observation_path,
        conditions=condition_path,
        key=key,
        optim=optim,
        target=target_posterior,
        num_steps=config.optimization.total_steps,
        run_tracker=tracker,
        observations_per_step=config.observations_per_step,
        samples_per_context=config.samples_per_context,
        time_limit_s=config.optimization.time_limit_s,
    )

    # run sample again for testing purposes
    n_context = 100
    s_per_context = int(test_samples / n_context)
    theta_q, log_q_theta, x_q, log_q_x_path, _ = (
        fitted_approximation.joint_sample_and_log_prob(
            observation_path, None, key, n_context, s_per_context
        )
    )

    flat_theta_q = jax.tree_util.tree_map(lambda x: jnp.ravel(x), theta_q)
    return (
        flat_theta_q,
        (tracker, x_q, fitted_approximation, opt_state),
    )


@inference_method
def run_buffered_vi[
    ParticleT: seqjtyping.Particle,
    InitialParticleT: tuple[seqjtyping.Particle, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Particle, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](
    target_posterior: BayesianSequentialModel[
        ParticleT,
        InitialParticleT,
        TransitionParticleHistoryT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        HyperParametersT,
    ],
    hyperparameters: HyperParametersT,
    key: jaxtyping.PRNGKeyArray,
    observation_path: ObservationT,
    condition_path: ConditionT | None = None,
    test_samples: int = 1000,
    config: BufferedVIConfig = BufferedVIConfig(),
    tracker: typing.Any = None,
) -> tuple[InferenceParametersT, typing.Any]:
    sequence_length = observation_path.batch_shape[0]

    approximation = build_approximation(
        config,
        sequence_length,
        target_posterior,
        key,
    )

    if isinstance(config.optimization, AdamOpt):
        optim = optax.apply_if_finite(
            optax.adam(config.optimization.lr), max_consecutive_errors=100
        )
    elif isinstance(config.optimization, CosineOpt):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.optimization.peak_lr,
            peak_value=config.optimization.peak_lr,
            warmup_steps=config.optimization.warmup_steps,
            decay_steps=config.optimization.decay_steps,
            end_value=config.optimization.end_lr,
        )

        optim = optax.apply_if_finite(
            optax.adam(learning_rate=schedule), max_consecutive_errors=100
        )

    if config.pre_training_steps > 0:
        approximation, _ = train.train(
            model=approximation,
            observations=observation_path,
            conditions=condition_path,
            key=key,
            optim=optim,
            target=target_posterior,
            num_steps=config.pre_training_steps,
            run_tracker=tracker,
            observations_per_step=config.observations_per_step,
            samples_per_context=config.samples_per_context,
            pre_train=True,
        )

    fitted_approximation, opt_state = train.train(
        model=approximation,
        observations=observation_path,
        conditions=condition_path,
        key=key,
        optim=optim,
        target=target_posterior,
        num_steps=config.optimization.total_steps,
        run_tracker=tracker,
        observations_per_step=config.observations_per_step,
        samples_per_context=config.samples_per_context,
        time_limit_s=config.optimization.time_limit_s,
    )

    # run sample again for testing purposes
    n_context = 100
    s_per_context = int(test_samples / n_context)
    (
        theta_q,
        log_q_theta,
        x_q,
        log_q_x_path,
        (approx_start, theta_mask, y_batch, c_batch),
    ) = fitted_approximation.joint_sample_and_log_prob(
        observation_path, condition_path, key, n_context, s_per_context
    )

    flat_theta_q = jax.tree_util.tree_map(lambda x: jnp.ravel(x), theta_q)
    return (
        flat_theta_q,
        (approx_start, x_q, tracker, fitted_approximation, opt_state),
    )
