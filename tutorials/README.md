# Tutorials

Introductory examples for `pioran_periodicity`.

## [`01_simulate_and_fit.py`](01_simulate_and_fit.py)

The fastest way in: simulate a short light curve, fit a DRW noise model
against a DRW+sine alternative, and read off the Bayes factor.

```
conda run -n <env> python tutorials/01_simulate_and_fit.py
```

## [`02_full_workflow.ipynb`](02_full_workflow.ipynb)

The full workflow, end to end, as a notebook:

1. simulate two light curves from the same red-noise process — one plain,
   one with an injected periodic signal;
2. an OBPL basis-function approximation-accuracy diagnostic
   (`psd_approximation_error` vs. `n_components`);
3. define DRW, CARMA(2,1) and OBPL noise-model families, each with a
   `"plain"` and a `"+sine"` variant sharing identical noise priors;
4. fit all 12 models with nested sampling (`fit_family`);
5. diagnostics: posterior corner plots and posterior-predictive checks;
6. model comparison: log10 Bayes factors for periodicity, by noise family,
   for both light curves — including a worked note on why a *too-wide*
   period prior can make even a noise-only light curve look periodic (the
   "look-elsewhere effect"), and how the notebook guards against it.

Open it with:

```
conda run -n <env> jupyter lab tutorials/02_full_workflow.ipynb
```

or re-run it end-to-end from the command line:

```
conda run -n <env> jupyter nbconvert --to notebook --execute --inplace \
    tutorials/02_full_workflow.ipynb
```

Both tutorials need `pip install -e ".[simulation]"` for the simulator and
a working pioranpy for the GP likelihood; `02_full_workflow.ipynb` also
needs `corner` (an `ultranest` dependency, already installed alongside it)
and a Jupyter kernel for the environment.

More tutorials (real-data walkthrough, PSD-approximation deep dive) are
planned — contributions welcome, see [CONTRIBUTING.md](../CONTRIBUTING.md).
