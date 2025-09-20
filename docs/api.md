# API Reference

The following sections document the public API for SeqJAX.

## Models

### Core interfaces

::: seqjax.model.base

### Simulation

::: seqjax.model.simulate

### Evaluation

::: seqjax.model.evaluate

## Inference

### Inference interfaces

Utilities that standardise the inputs and outputs of inference routines. The
[particle filtering](tutorials/particle-filtering.md),
[Bayesian MCMC](tutorials/bayesian-mcmc.md), and
[variational inference](tutorials/variational-inference.md) tutorials provide
hands-on examples built on top of these interfaces.

::: seqjax.inference.interface

### Kalman filtering

::: seqjax.inference.kalman

### Particle filtering

::: seqjax.inference.particlefilter

### MCMC kernels

::: seqjax.inference.mcmc

### Particle MCMC

::: seqjax.inference.pmcmc

### Stochastic gradient Langevin dynamics

::: seqjax.inference.sgld

### Variational inference

::: seqjax.inference.vi
