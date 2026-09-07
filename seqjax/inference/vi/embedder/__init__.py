"""Embedding interfaces, implementations, and configuration factories."""

from .embedder import (
    Conv1DEmbedder,
    PositionalEmbedder,
    RNNEmbedder,
    TransformerEmbedder,
    WindowEmbedder,
)
from .interface import Embedder, LatentContext, LatentContextDims, SequenceAggregator
from .norm import NormalizationConfig
from .registry import (
    BiRNNEmbedder,
    Conv1DEmbedderConfig,
    EmbedderConfig,
    EmbedderName,
    LongContextEmbedder,
    PassthroughEmbedder,
    PositionalEmbedderConfig,
    ShortContextEmbedder,
    TransformerEmbedderConfig,
    build_embedder,
    embedder_registry,
)

__all__ = [
    "BiRNNEmbedder",
    "Conv1DEmbedder",
    "Conv1DEmbedderConfig",
    "Embedder",
    "EmbedderConfig",
    "EmbedderName",
    "LatentContext",
    "LatentContextDims",
    "LongContextEmbedder",
    "NormalizationConfig",
    "PassthroughEmbedder",
    "PositionalEmbedder",
    "PositionalEmbedderConfig",
    "RNNEmbedder",
    "SequenceAggregator",
    "ShortContextEmbedder",
    "TransformerEmbedder",
    "TransformerEmbedderConfig",
    "WindowEmbedder",
    "build_embedder",
    "embedder_registry",
]
