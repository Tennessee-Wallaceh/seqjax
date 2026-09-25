"""

"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jaxtyping import PRNGKeyArray, Array

from seqjax.inference.interface import InferenceDataset
import seqjax.model.typing as seqjtyping



def sample_sequence_minibatch[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    NumSequence: seqjtyping.NumSequence,
    NumSequenceSamples: seqjtyping.NumSequence,
    SequenceLength: seqjtyping.SequenceLength,
](
    dataset: InferenceDataset[
        ObservationT, 
        ConditionT,
        NumSequence,
        SequenceLength,
    ],
    key: PRNGKeyArray,
    num_sequence_minibatch: NumSequenceSamples = 1,
) -> tuple[
    seqjtyping.Batched[
        ObservationT, NumSequenceSamples, SequenceLength
    ],
    seqjtyping.Batched[
        ConditionT, NumSequenceSamples, SequenceLength
    ]    
]:

    if num_sequence_minibatch <= 0:
        raise ValueError(
            "num_sequence_minibatch must be positive. "
            f"Received {num_sequence_minibatch}."
        )
    if num_sequence_minibatch > dataset.num_sequences:
        raise ValueError(
            "num_sequence_minibatch cannot exceed dataset.num_sequences. "
            f"Received num_sequence_minibatch={num_sequence_minibatch}, "
            f"dataset.num_sequences={dataset.num_sequences}."
        )
    
    minibatch_index = jrandom.choice(
        key,
        dataset.num_sequences,
        shape=(num_sequence_minibatch,),
        replace=False,
    )

    sampled_observations = jax.tree_util.tree_map(
        lambda leaf: leaf[minibatch_index],
        dataset.observations,
    )
    sampled_conditions = jax.tree_util.tree_map(
        lambda leaf: leaf[minibatch_index],
        dataset.conditions,
    )

    return sampled_observations, sampled_conditions


def sample_buffered_subsequence[
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    SequenceLength: seqjtyping.SequenceLength,
    SampleLength: seqjtyping.SampleLength,
](
    key: PRNGKeyArray, 
    sequence_length: SequenceLength, 
    sample_length: seqjtyping.SampleLength, 
    buffer_length: seqjtyping.BufferLength, 
    observation_path: seqjtyping.Batched[ObservationT, SequenceLength],
    condition: seqjtyping.Batched[ConditionT, SequenceLength],
) -> tuple[
    Array,
    seqjtyping.Batched[ObservationT, SampleLength],
    seqjtyping.Batched[ConditionT, SampleLength],
    Array,
]:
    if sample_length <= 2 * buffer_length:
        raise ValueError("Sample length must contain at least one target observation.")
    
    if sample_length > sequence_length:
        raise ValueError("Sample length exceeds sequence length.")
    
    # usually the latent prior length is 1
    # if it exceeds this, then we should sample a shorter sequence of observations
    # to maintain the same target batch + buffer length
    batch_length = sample_length - 2 * buffer_length
    pad_length = batch_length - 1

    # each sample will come from the data replicated onto each device
    padded_start_ix = jrandom.randint(
        key, (), 0, sequence_length + pad_length
    )

    # where the buffer would like to start, may fall outside of data
    buffer_start = padded_start_ix - pad_length - buffer_length

    # clip into possible starts for the latent approximation
    approx_start = jnp.clip(
        buffer_start, min=0, max=sequence_length - sample_length
    )

    # construct the mask for theta
    batch_start = padded_start_ix - pad_length  # may be negative (left padding)
    sample_index = approx_start + jnp.arange(sample_length)  # data indices covered by `samples`
    theta_mask = (sample_index >= batch_start) & (sample_index < batch_start + batch_length)
    theta_mask = theta_mask & (sample_index >= 0) & (sample_index < sequence_length)

    samples = jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=approx_start,
            slice_size=sample_length,
        ),
        observation_path,
    )
    csamples = jax.tree_util.tree_map(
        partial(
            jax.lax.dynamic_slice_in_dim,
            start_index=approx_start,
            slice_size=sample_length,
        ),
        condition,
    )

    return approx_start, samples, csamples, theta_mask
