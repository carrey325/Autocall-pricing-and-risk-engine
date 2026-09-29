# Contributing

Changes must preserve the composable product model and reproducible numerical
behavior. Runtime code lives in `src/structured_products`; install the package
before running tests so Python imports the same package users will install.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell activate with `.venv\Scripts\Activate.ps1` instead.
When moving an existing editable checkout from the old flat layout, recreate
the environment and remove stale root-level `*.egg-info` build metadata.

## Run tests

```bash
python -m coverage run -m pytest
python -m coverage report
```

## Build and verify the installable artifacts

```bash
python -m build
python -m twine check dist/*
python tools/verify_distribution.py --dist dist
```

Use a fresh output directory when rebuilding another version. The verifier
audits wheel/sdist contents and installs each into a separate temporary virtual
environment outside the repository. It exercises the Python API, all six
bundled examples, the console and module CLI, and dependency compatibility.
This requires network access for dependencies. CI repeats it on Linux, Windows,
and macOS with Python 3.10 and 3.13, and retains the validated build as a workflow
artifact. There is no upload-to-PyPI or automatic release workflow.

`src/structured_products/_version.py` is the only version source. Package data
under `src/structured_products/data` mirrors the six editable JSON examples in
`configs/`; tests enforce equality. `ResolvedConfig.to_dict()` is for inspecting
resolved contracts, not for recreating the template-based input schema.

## Contribution guidelines

- Never add or execute legacy notebooks or employer-specific source.
- Express marketing variants with feature and schedule data, not new engines.
- Add hand-calculated payoff tests before extending a numerical engine.
- Use common random numbers for Monte Carlo comparisons.
- Document and justify numerical tolerances in `VALIDATION.md`.
- Do not commit generated files such as `__pycache__`, `.DS_Store`, build
  outputs, or notebook checkpoints.
