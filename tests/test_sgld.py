import jax.numpy as jnp
import jax.random as jrandom

from seqjax.inference.sgld import SGLDConfig, run_sgld
from seqjax.model.ar import ARParameters


def dummy_grad(params: ARParameters, key: jrandom.PRNGKey) -> ARParameters:
    del key
    return ARParameters(
        ar=jnp.ones_like(params.ar),
        observation_std=jnp.zeros_like(params.observation_std),
        transition_std=jnp.zeros_like(params.transition_std),
    )


def test_run_sgld_basic() -> None:
    params = ARParameters()
    cfg = SGLDConfig(step_size=0.1, num_iters=4)
    samples = run_sgld(dummy_grad, jrandom.PRNGKey(0), params, config=cfg)
    assert samples.ar.shape == (cfg.num_iters,)

