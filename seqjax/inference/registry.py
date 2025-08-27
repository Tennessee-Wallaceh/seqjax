""" """

import typing
from dataclasses import dataclass
from functools import partial

from seqjax.inference import InferenceMethod, pmcmc, mcmc, particlefilter, sgld

inference_functions = {"NUTS": mcmc.run_bayesian_nuts}


@dataclass
class NUTSInference:
    method: typing.Literal["NUTS"]
    config: mcmc.NUTSConfig


InferenceConfig = NUTSInference


def build_inference(i_config: InferenceConfig, target_model) -> InferenceMethod:
    return partial(inference_functions[i_config.method], config=i_config.config)
