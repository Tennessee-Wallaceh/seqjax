import typing


class VISamplingKwargs(typing.TypedDict):
    context_samples: int
    samples_per_context: int
    num_sequence_minibatch: int


class VISampleConfig(typing.Protocol):
    def training_sampling_kwargs(self, *, loss_label: str) -> VISamplingKwargs: ...

    def evaluation_sampling_kwargs(self, *, test_samples: int) -> VISamplingKwargs: ...
