#!/usr/bin/env python
import argparse
import json
from logging import config
import sys
from typing import Dict, Any, Tuple, List
from seqjax.inference import vi
from seqjax.inference import registry as inference_registry
from experiments.core import ExperimentConfig, run_experiment
from seqjax.model import registry as model_registry
from seqjax import io

codes: Dict[str, Dict[str, Any]] = {
    "LR": {
        "1e-2": 1e-2,
        "1e-3": 1e-3,
        "1e-4": 1e-4,
    },
    "MC": {
        "1": 1,
        "10": 10,
    },
    "BS": {  # Minibatch size
        "1": 1,
        "10": 10,
    },
    "PT": {  # Pre-training steps
        "N": 0,
        "Y": 5000,
    },
    "B": {  # Buffer length
        "5": 5,
        "20": 20,
    },
    "M": {  # Batch length
        "5": 5,
        "20": 20,
    },
    "EMB": {  # Embedding
        "LC": {"type": "long_context", "window": 5},
        "SC": {"type": "short_context", "window": 2},
        "BiRNN": {"type": "birnn", "hidden_dim": 5},
    },
}

factor_names: Dict[str, str] = {
    "LR": "learning_rate",
    "MC": "mc_samples",
    "BS": "minibatch_size",
    "PT": "pretrain_steps",
    "B": "buffer_len",
    "M": "batch_len",
    "EMB": "embedding",
}


default_codes: List[str] = ["LR-1e-3", "MC-1", "BS-1", "PT-N", "B-5", "M-5", "EMB-SC"]


def parse_token(token: str) -> Tuple[str, Any]:
    if "-" not in token:
        raise argparse.ArgumentTypeError(
            f"Bad code '{token}'. Expected PREFIX-SUFFIX like 'LR-1e-3'."
        )
    prefix, suffix = token.split("-", 1)
    if prefix not in codes:
        raise argparse.ArgumentTypeError(
            f"Unknown factor '{prefix}'. Valid factors: {', '.join(codes.keys())}"
        )
    sub = codes[prefix]
    if suffix not in sub:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{suffix}' for factor '{prefix}'. Choices: {', '.join(sub.keys())}"
        )
    return prefix, sub[suffix]


def normalize_tokens(raw: List[str]) -> List[str]:
    tokens: List[str] = []
    for r in raw:
        tokens.extend([t.strip() for t in r.split(",") if t.strip()])
    return tokens


def resolve_config(tokens: List[str]) -> Dict[str, Any]:
    # Start from defaults
    resolved: Dict[str, Any] = {}
    for tok in default_codes:
        k, v = parse_token(tok)
        resolved[factor_names.get(k, k)] = v

    # Apply user overrides (last-one-wins per factor)
    seen = set()
    for tok in tokens:
        k, v = parse_token(tok)
        key = factor_names.get(k, k)
        resolved[key] = v
        seen.add(k)
    return resolved


def list_all_codes() -> str:
    lines = []
    for k, sub in codes.items():
        vals = ", ".join(f"{k}-{s}" for s in sub.keys())
        lines.append(f"{k}: {vals}")
    return "\n".join(lines)


def parse(argv=None):
    p = argparse.ArgumentParser(
        description="Parse shorthand config codes like LR-1e-3,MC-10,EMB-BiRNN."
    )
    p.add_argument(
        "-c",
        "--cfg",
        action="append",
        default=[],
        help="Codes (repeatable or comma-separated), e.g. -c LR-1e-4,MC-10 -c EMB-BiRNN",
    )
    p.add_argument(
        "--show-codes",
        action="store_true",
        help="Print all valid shorthand codes and exit.",
    )

    p.add_argument(
        "-s", "--fit_seed", type=int, required=True, help="Random seed (int)."
    )
    p.add_argument(
        "-d", "--data_seed", type=int, required=True, help="Data seed (int)."
    )
    p.add_argument("-n", "--name", type=str, required=True, help="Experiment name.")

    args = p.parse_args(argv)

    if args.show_codes:
        print(list_all_codes())
        return

    tokens = normalize_tokens(args.cfg)
    try:
        cfg = resolve_config(tokens)
    except argparse.ArgumentTypeError as e:
        p.error(str(e))

    return cfg, args.fit_seed, args.data_seed, args.name


class ResultProcessor:
    def process(
        self,
        wandb_run,
        experiment_config,
        param_samples,
        extra_data,
        x_path,
        y_path,
        condition,
    ) -> None:
        approx_start, x_q, run_tracker, fitted_approximation = extra_data

        # save final model
        io.save_model_artifact(
            wandb_run,
            f"{wandb_run.name}-fitted-approximation",
            fitted_approximation,
        )

        checkpoint_samples = getattr(run_tracker, "checkpoint_samples", [])
        if checkpoint_samples:
            io.save_packable_artifact(
                wandb_run,
                f"{wandb_run.name}_checkpoint_samples",
                "checkpoint_samples",
                [
                    (
                        f"samples_{i}",
                        samples,
                        {"elapsed_time_s": float(elapsed_time_s)},
                    )
                    for i, (elapsed_time_s, samples) in enumerate(checkpoint_samples)
                ],
            )


if __name__ == "__main__":
    out = parse()
    if out is None:
        sys.exit(0)

    SEQUENCE_LENGTH = 10000
    TEST_SAMPLES = 10000

    config, fit_seed, data_seed, experiment_name = out
    print(json.dumps(config, indent=2), fit_seed, data_seed)

    if config["embedding"]["type"] == "long_context":
        embedder = vi.run.LongContextEmbedder()
    elif config["embedding"]["type"] == "short_context":
        embedder = vi.run.ShortContextEmbedder()
    elif config["embedding"]["type"] == "birnn":
        embedder = vi.run.BiRNNEmbedder()
    else:
        raise ValueError(f"Unknown embedder type: {config['embedding']['type']}")

    buffviconf = inference_registry.BufferVI(
        "buffer-vi",
        vi.BufferedVIConfig(
            optimization=vi.run.AdamOpt(
                lr=config["learning_rate"],
                total_steps=50000,
                time_limit_s=60 * 60 * 2,  # 2 hour limit
            ),
            buffer_length=config["buffer_len"],
            batch_length=config["batch_len"],
            parameter_field_bijections={
                "std_log_vol": "softplus",
                "mean_reversion": "softplus",
                "long_term_vol": "softplus",
            },
            control_variate=False,
            embedder=embedder,
            pre_training_steps=config["pretrain_steps"],
            observations_per_step=config["minibatch_size"],
            samples_per_context=config["mc_samples"],
        ),
    )

    experiment_config = ExperimentConfig(
        data_config=model_registry.DataConfig(
            target_model_label="simple_stochastic_vol",
            generative_parameter_label="base",
            sequence_length=SEQUENCE_LENGTH,
            seed=data_seed,
        ),
        test_samples=TEST_SAMPLES,
        fit_seed=fit_seed,
        inference=buffviconf,
    )

    output = run_experiment(
        experiment_name,
        experiment_config,
        result_processor=ResultProcessor(),
    )
