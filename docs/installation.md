# Installation

## Prerequisite: pioranpy / Julia

`pioran_periodicity` builds its GP kernels via
[pioranpy](https://github.com/mlefkir/pioranpy), a Python wrapper around
[Pioran.jl](https://github.com/mlefkir/Pioran.jl) using `juliacall`/`PythonCall`.

Installing `pioranpy` (via pip) triggers a Julia install and `Pioran.jl`
precompilation on first import. This needs network access and can take a
few minutes the first time; there is no pure-Python fallback — every module
that constructs a kernel (`pioran_periodicity.kernels`) imports pioranpy.

## Install the package

```bash
conda create -n pioran-periodicity python=3.11
conda activate pioran-periodicity
pip install -e ".[simulation,test]"
```

Optional extras:

- `simulation` — pulls in [stingray](https://stingray.readthedocs.io/) for
  `pioran_periodicity.simulate` (TK95 light-curve simulation).
- `test` — `pytest`, for running `tests/`.
- `dev` — `black`, `flake8`, `isort`, `pre-commit`.
- `docs` — `mkdocs-material`, `mkdocstrings`.

## Verify

```bash
conda run -n pioran-periodicity python -c \
    "import pioran_periodicity as pp; print(pp.__version__)"
conda run -n pioran-periodicity pytest tests/
```
