# Contributing

## Dev setup

```
conda create -n pioran-periodicity python=3.11
conda activate pioran-periodicity
pip install -e ".[simulation,test,dev]"
pre-commit install
```

### pioranpy / Julia prerequisite

The package depends on [pioranpy](https://github.com/mlefkir/pioranpy), a
Python wrapper around [Pioran.jl](https://github.com/mlefkir/Pioran.jl) via
`juliacall`/`PythonCall`. Installing `pioranpy` (via pip) triggers a Julia
install and `Pioran.jl` precompilation on first import — this can take a few
minutes and needs network access. There is no way around this: every module
that builds a kernel (`pioran_periodicity.kernels`) imports pioranpy.

## Running tests

```
conda run -n pioran-periodicity pytest tests/
```

The suite is pure-package (no simulation-study or real-data driver scripts
are exercised); it does need a working pioranpy since kernel construction
goes through Pioran.jl.

## Style

- **Formatting:** `black .`
- **Import order:** `isort .`
- **Linting:** `flake8`
- All three run automatically via `pre-commit` if installed; CI enforces
  them independently of the test job.

## Adding a driver script

Scripts under `scripts/` and `paper/` are thin CLIs over the package's
public API. New scripts should:

- take explicit `--data-dir`/`--results-dir`/`--out-dir` arguments rather
  than hardcoding paths;
- be resumable where practical (skip existing output files);
- document expected input format and outputs in the module docstring and
  in `scripts/README.md` / `paper/README.md`.
