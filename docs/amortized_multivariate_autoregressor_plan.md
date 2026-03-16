# Implementation plan: `AmortizedMultivariateAutoregressor`

This plan mirrors the current univariate implementation in `seqjax/inference/vi/autoregressive.py` and compares two implementation approaches.

## Goals

- Add a production `AmortizedMultivariateAutoregressor` that follows the control flow and interface style of `AmortizedUnivariateAutoregressor`.
- Keep behavior compatible with `AutoregressiveApproximation.sample_and_log_q`.
- Preserve numerical stability and provide informative error paths for shape mismatches.
- Use protocol-based typing for network outputs so we can swap MLP backends safely.

## Current univariate pattern to follow

The univariate implementation establishes a clear pattern:

1. Build an amortizer network in `__init__` with input dims from
   - lag history,
   - availability flags,
   - parameter embedding,
   - sequence context,
   - condition embedding.
2. In `conditional(...)`, concatenate feature blocks into one vector.
3. Draw standard normal noise (`z`).
4. Map network outputs to distribution parameters (`loc`, `scale`).
5. Reparameterize sample and compute `log_q_x`.

The multivariate version should preserve these five steps, only changing parameterization from scalar normal to multivariate normal.

## Option A (recommended): Full-covariance parameterization via Cholesky factor

### Summary

Predict both:

- `loc` with shape `(x_dim,)`
- unconstrained lower-triangular parameters for a Cholesky factor `L` with shape `(x_dim, x_dim)`

Then:

- use `diag = softplus(raw_diag) + eps` for positive diagonal
- form `L`
- sample with `x = loc + L @ z`
- evaluate `log_q_x` from the Gaussian implied by `(loc, L)`

### Why this is closest to the univariate version

Univariate `scale` generalizes to multivariate `L`; both are direct reparameterization objects. This keeps code shape and reasoning almost identical.

### Pros

- Maximum expressiveness (captures correlations in `q`).
- Conceptually aligned with the previously sketched, commented class.
- Best fit when latent dimensions are strongly coupled.

### Cons

- More parameters (`O(x_dim^2)`).
- More expensive and potentially less numerically stable if not carefully constrained.

### Error-handling recommendations

Prefer explicit errors when assumptions fail:

- if `x_dim <= 0`: raise `ValueError("x_dim must be positive for multivariate autoregressor")`
- if amortizer output size does not match expected triangular packing size: raise `RuntimeError` with expected/actual sizes.
- if reconstructed matrix has non-finite entries: raise `FloatingPointError` with step context.

## Option B: Diagonal covariance parameterization

### Summary

Predict:

- `loc` with shape `(x_dim,)`
- per-dimension unconstrained scales with shape `(x_dim,)`

Then:

- `scale = softplus(raw_scale) + eps`
- sample elementwise `x = loc + scale * z`
- compute `log_q_x` as sum of univariate logpdfs.

### Pros

- Simpler implementation and debugging.
- Lower compute and memory costs.
- Numerically robust, especially for larger latent dimensions.

### Cons

- Cannot represent cross-dimension correlations.
- Could degrade approximation quality for correlated latent states.

### When to choose

- If we need a fast initial implementation that is easy to validate.
- If model latents are near-conditionally independent or dimensions are high.

## Recommended rollout strategy

1. Implement Option B first as a minimal parity baseline.
2. Add Option A behind a small configuration switch (e.g., `covariance="diag" | "full"`).
3. Keep default as `diag` until tests and benchmarks validate `full` robustness.

This gives us a stable first milestone while still enabling richer approximation later.

## Protocol-based typing proposal

Introduce a protocol that captures the amortizer call contract (instead of hard-coding concrete `eqx.nn.MLP | ResNetMLP` unions).

Example typing direction:

- `class VectorToVector(Protocol):`
  - `def __call__(self, x: Float[Array, "input_dim"]) -> Float[Array, "output_dim"]: ...`

Then annotate amortizer fields as protocol types:

- `amortizer_mlp: VectorToVector`

Benefits:

- Easier backend swaps (ResNet MLP, plain MLP, custom modules).
- Cleaner mypy behavior than large concrete unions.
- Better supports test doubles/mocks.

## Concrete implementation checklist

- [ ] Add `AmortizedMultivariateAutoregressor` class with the same constructor pattern as univariate.
- [ ] Reuse the existing context assembly in `conditional(...)` with strict shape assertions.
- [ ] Implement `diag` covariance path first (Option B).
- [ ] Add a second `full` covariance path (Option A), using packed-triangular output and stable diagonal transform.
- [ ] Register the class in inference registry where multivariate latent classes are selected.
- [ ] Add tests for:
  - [ ] output/sample shapes
  - [ ] finite log densities
  - [ ] deterministic behavior under fixed key
  - [ ] explicit errors for malformed dimensions
- [ ] Run mypy and pytest coverage for new path.

## Validation plan

- Unit tests for shape and log-density finiteness at small dimensions (`x_dim=2,3`).
- Regression test against current univariate path when `x_dim=1` (sanity equivalence, up to parameterization differences).
- Short experiment in `experiments/` comparing ELBO trend for `diag` vs `full` on a small correlated-latent synthetic model.

## Open decision points

1. Should registry expose covariance choice as a user-facing config flag now, or keep internal until stabilized?
2. Should we support a low-rank-plus-diagonal path as a midpoint between options A and B?
3. If `x_dim` is large, should we automatically fall back to `diag` unless explicitly overridden?
