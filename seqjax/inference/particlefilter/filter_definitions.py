from __future__ import annotations

from functools import partial

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
)

from .base import GeneralSequentialImportanceSampler
from .resampling import conditional_resample, gumbel_resample_from_log_weights


class BootstrapParticleFilter(
    GeneralSequentialImportanceSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ]
):
    """Classical bootstrap particle filter."""

    def __init__(
        self,
        target: SequentialModel[
            ParticleType, ObservationType, ConditionType, ParametersType
        ],
        num_particles: int,
    ) -> None:
        super().__init__(
            target=target,
            proposal=target.transition,
            resampler=partial(
                conditional_resample,
                resampler=gumbel_resample_from_log_weights,
                esse_threshold=0.5,
            ),
            num_particles=num_particles,
        )
