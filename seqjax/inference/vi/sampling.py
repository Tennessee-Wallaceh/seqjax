import typing
import seqjax.model.typing as seqjtyping

class VISamplingKwargs(typing.TypedDict):
    context_samples: seqjtyping.NumBatches
    samples_per_context: seqjtyping.NumMonteCarlo
    num_sequence_minibatch: seqjtyping.NumSequence


class VISampleConfig(typing.Protocol):
    def training_sampling_kwargs(self, *, loss_label: str) -> VISamplingKwargs: ...

    def evaluation_sampling_kwargs(self, *, test_samples: int) -> VISamplingKwargs: ...
