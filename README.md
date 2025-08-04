# SeqJAX

SeqJAX provides utilities for building sequential probabilistic models with [JAX](https://github.com/google/jax). The library encourages a functional style: models are composed of `Prior`, `Transition` and `Emission` classes which operate on simple dataclasses for particles, observations and parameters. Runtime interface checks ensure that these components implement the correct methods and signatures, reducing boilerplate errors. The three components are grouped together in a ``SequentialModel`` definition.

SeqJAX ships with a handful of toy models demonstrating this approach: an AR(1) process, a stochastic volatility model and a multidimensional linear Gaussian state space model.

## Installation

SeqJAX requires Python 3.12 or later. Only installation from source is currently available:

```bash
pip install git+https://github.com/bayesianshift/seqjax.git
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
latent_path, observation_path, latent_hist, obs_hist = simulate.simulate(
    key, model, condition=None, parameters=parameters, sequence_length=5
)
print(observation_path)
```

SeqJAX will check at runtime that `AR1Target` and its components implement the required interface. When extending the library, these checks help catch mistakes such as missing `sample` or `log_prob` implementations.

## Documentation

The library's API reference can be built using [MkDocs](https://www.mkdocs.org/) together with the [mkdocstrings](https://mkdocstrings.github.io/) plugin. After installing the optional dependencies

```bash
pip install mkdocs mkdocstrings[python]
```

add ``mkdocstrings`` to your ``mkdocs.yml`` and include objects with the ``:::`` syntax:

```markdown
::: seqjax.model.LinearGaussianSSM
```

Run ``mkdocs serve`` to preview the documentation locally or ``mkdocs build`` to generate a site.
