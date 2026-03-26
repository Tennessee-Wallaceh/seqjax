# SeqJAX

SeqJAX provides utilities for building sequential probabilistic models with [JAX](https://github.com/google/jax). The library encourages a functional style: models are composed of `Prior`, `Transition` and `Emission` classes which operate on simple dataclasses for particles, observations and parameters. Runtime interface checks ensure that these components implement the correct methods and signatures, reducing boilerplate errors. The three components are grouped together in a ``SequentialModel`` definition.

SeqJAX ships with a handful of toy models demonstrating this approach: an AR(1) process, a stochastic volatility model and a multidimensional linear Gaussian state space model.

## Installation

SeqJAX requires Python 3.13 or later. Only installation from source is currently available:

```bash
pip install git+https://github.com/Tennessee-Wallaceh/seqjax.git
```

## Example

The `seqjax.model.ar` module contains a small autoregressive example. The snippet below defines a model using these components and simulates a short path of data.

```python
import jax.random as jrandom
from seqjax.model.ar import AR1Target, ARParameters
from seqjax import simulate

# Model parameters and target
parameters = ARParameters(
    ar=0.8,            # autoregressive coefficient
    observation_std=1.0,
    transition_std=0.5,
)
model = AR1Target()

key = jrandom.key(0)

# Simulate a sequence of length 5
latent_path, observation_path = simulate.simulate(
    key, model, condition=None, parameters=parameters, sequence_length=5
)
print(observation_path)
```

SeqJAX will check at runtime that `AR1Target` and its components implement the required interface. When extending the library, these checks help catch mistakes such as missing `sample` or `log_prob` implementations.


## Design

- A `registry` converts static config (a dataclass) to objects
- A `run` submodule defines an interface between static config and models, building appropriate objects and using them to produce posterior samples
- The `cli` submodule is a command line interface for targeting data with inference configured by command line options. It relies on a remmote backend for storing+loading run outputs. The idea is to convert flat strings into config understood by a `registry`.

## Local/offline experiment storage

The CLI `run` command now supports:

- `--storage-mode wandb` (default): standard online W&B runs/artifacts.
- `--storage-mode wandb-offline`: store run metadata and artifacts on local disk without uploading.
- `--local-root <path>`: base directory used for local W&B files when offline mode is enabled.

Typical local layout in offline mode looks like:

```
<local-root>/
  wandb/
    offline-run-*/
      files/
      logs/
      run-*.wandb
```

To sync an offline run later, point `wandb` at the created offline run directory:

```bash
wandb sync <local-root>/wandb/offline-run-*
```

Use the normal W&B login flow before syncing if you want those runs uploaded to a remote project.


## Supported data usage patterns

SeqJAX currently supports multiple data loading workflows from the CLI.

### 1) Simulated data + online W\&B artifacts (default)

This is the default pattern. Data is generated or fetched through W\&B artifacts and inference logs online.

```bash
seqjax run my-project   --model ar   --parameters base   --sequence-length 256   --num-sequences 1   --data-seed 0   --fit-seed 1   --inference buffer-vi
```

### 2) Simulated data + local/offline storage

Use offline mode to keep data and run outputs local.

```bash
seqjax run my-project   --storage-mode wandb-offline   --local-root ./wandb   --model ar   --parameters base   --sequence-length 256   --num-sequences 1   --data-seed 0   --fit-seed 1   --inference buffer-vi
```

### 3) Prepared local dataset by explicit dataset reference

First, preprocess and save data under a dataset name in `local_root/datasets/<dataset_name>/`.
The helper script in `experiments/process_prepared_dataset.py` demonstrates this end-to-end.

Then run with `--data-source prepared-local` and optionally set `--prepared-dataset-name` to override
`DataConfig.dataset_name`.

```bash
seqjax run my-project   --data-source prepared-local   --local-root ./wandb   --prepared-dataset-name aicher_stochastic_vol-real-v1   --model aicher_stochastic_vol   --parameters base   --sequence-length 256   --num-sequences 1   --data-seed 0   --fit-seed 1   --inference buffer-vi
```

Prepared datasets include a manifest (`dataset_manifest.json`) that is validated before loading.
This adds safety checks for:
- model label compatibility,
- sequence length,
- number of sequences,
- and expected dataset name.
