"""
Embedding configurations
"""
import typing
from dataclasses import dataclass, field
from functools import partial

import equinox as eqx
import jax.random as jrandom
import jaxtyping

import seqjax.model.typing as seqjtyping
from seqjax.inference.vi.embedder import embedder
from seqjax.inference.vi.embedder import aggregation
from seqjax.model import interface as model_interface



PositionMode = typing.Literal["sample", "sequence"]


EmbedderName = typing.Literal[
    "short-window", 
    "long-window", 
    "bi-rnn", 
    "passthrough", 
    "conv1d", 
    "transformer", 
    "positional"
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
    condition_on_parameters: bool = False


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

def build_embedder(
    embedder_config: EmbedderConfig,
    target: model_interface.SequentialModelProtocol,
    inference_parameter_cls: type[seqjtyping.Parameters],
    sequence_length: int,
    sample_length: int,
    embedding_key: jaxtyping.PRNGKeyArray,
) -> embedder.Embedder:
    embed: embedder.Embedder
    if isinstance(embedder_config, ShortContextEmbedder):
        embed = embedder.WindowEmbedder(
            target=target,
            parameter_cls=inference_parameter_cls,
            sample_length=sample_length,
            sequence_length=sequence_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
        )
    elif isinstance(embedder_config, PassthroughEmbedder):
        embed = embedder.WindowEmbedder(
            target=target,
            parameter_cls=inference_parameter_cls,
            sample_length=sample_length,
            sequence_length=sequence_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
        )
    elif isinstance(embedder_config, LongContextEmbedder):
        embed = embedder.WindowEmbedder(
            target=target,
            parameter_cls=inference_parameter_cls,
            sample_length=sample_length,
            sequence_length=sequence_length,
            prev_window=embedder_config.prev_window,
            post_window=embedder_config.post_window,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
        )
    elif isinstance(embedder_config, Conv1DEmbedderConfig):
        embed = embedder.Conv1DEmbedder(
            target=target,
            parameter_cls=inference_parameter_cls,
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
            target=target,
            parameter_cls=inference_parameter_cls,
            sample_length=sample_length,
            sequence_length=sequence_length,
            hidden=embedder_config.hidden_dim,
            aggregation_kind=embedder_config.aggregation_kind,
            position_mode=embedder_config.position_mode,
            n_pos_embedding=embedder_config.n_pos_embedding,
            condition_on_parameters=embedder_config.condition_on_parameters,
            key=embedding_key,
        )
    elif isinstance(embedder_config, TransformerEmbedderConfig):
        embed = embedder.TransformerEmbedder(
            target=target,
            parameter_cls=inference_parameter_cls,
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
            target=target,
            parameter_cls=inference_parameter_cls,
            sample_length=sample_length,
            sequence_length=sequence_length,
            n_pos_embedding=embedder_config.n_pos_embedding,
            position_mode=embedder_config.position_mode,
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder_config.label}")

    return embed