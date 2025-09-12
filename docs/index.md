# SeqJAX

SeqJAX provides utilities for building sequential probabilistic models with [JAX](https://github.com/google/jax). Models are composed of `Prior`, `Transition`, and `Emission` classes operating on simple dataclasses for particles, observations, and parameters. Runtime interface checks ensure that these components implement the correct methods and signatures, reducing boilerplate errors.
