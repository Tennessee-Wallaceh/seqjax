# Getting Started

This short tutorial demonstrates a simple autoregressive model using `seqjax`.

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

latent_path, observation_path, latent_hist, obs_hist = simulate.simulate(
    key, model, condition=None, parameters=parameters, sequence_length=5,
)
print(observation_path)
```
