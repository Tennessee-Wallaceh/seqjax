# SeqJAX

SeqJAX provides utilities for building sequential probabilistic models with [JAX](https://github.com/google/jax). The library encourages a functional style where models are composed of `Prior`, `Transition`, and `Emission` classes that operate on simple dataclasses for particles, observations, and parameters. Runtime interface checks ensure that these components implement the required methods and signatures, reducing boilerplate errors when extending or experimenting with new models.

## Feature overview

- **Composable building blocks.** Combine prior, transition, and emission components into a single `SequentialModel` definition.
- **Runtime interface validation.** Helpful checks catch mistakes such as missing `sample` or `log_prob` implementations while you iterate.
- **Demonstration models included.** Explore AR(1), stochastic volatility, and multidimensional linear Gaussian state space models to understand the provided abstractions.

## Installation

SeqJAX requires Python 3.13 or later. Install the latest version directly from the source repository:

```bash
pip install git+https://github.com/bayesianshift/seqjax.git
```

## Quick start

The `seqjax.model.ar` module contains a small autoregressive example. The snippet below defines a model using the building blocks above and simulates a short path of data.

```python
import jax.random as jrandom
from seqjax.model.ar import AR1Target, ARParameters
from seqjax import simulate

parameters = ARParameters(
    ar=0.8,
    observation_std=1.0,
    transition_std=0.5,
)
model = AR1Target()

key = jrandom.key(0)

latent_path, observation_path = simulate.simulate(
    key, model, condition=None, parameters=parameters, sequence_length=5,
)
print(observation_path)
```

SeqJAX checks at runtime that `AR1Target` and its components implement the required interface, making it easier to extend or customize the library. Once you are comfortable with simulation, explore the advanced inference workflows showcased in the tutorials below to analyse the resulting observation streams.

## Explore further

- Follow the [Getting Started tutorial](tutorials/getting-started.md) to walk through the autoregressive model step by step.
- Learn how to track filtering distributions with the [particle filtering walkthrough](tutorials/particle-filtering.md).
- Infer parameters jointly with latent states in the [Bayesian MCMC guide](tutorials/bayesian-mcmc.md).
- Optimise amortised posterior approximations using the [variational inference tutorial](tutorials/variational-inference.md).
- Browse the [API reference](api.md) for detailed documentation of available classes and functions.
