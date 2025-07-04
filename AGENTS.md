This project runs tests with `pytest` and type-checks with `mypy`.
Before committing any changes, run:

```
pip install .[dev]
pytest
mypy seqjax
```

Confirm that both the tests and the type-checking pass.

Always run new experiments in the experiments/ directory and verify that they run and produce sensible results.
