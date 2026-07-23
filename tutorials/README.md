# Tutorials

Introductory examples for `pioran_periodicity`. Start with
[`01_simulate_and_fit.py`](01_simulate_and_fit.py): simulate a short light
curve, fit a DRW noise model against a DRW+sine alternative, and read off
the Bayes factor.

Run it with:

```
conda run -n <env> python tutorials/01_simulate_and_fit.py
```

(needs `pip install -e ".[simulation]"` for the simulator, and a working
pioranpy for the GP likelihood).

More tutorials (real-data walkthrough, OBPL/CARMA model families, PSD
approximation diagnostics) are planned — contributions welcome, see
[CONTRIBUTING.md](../CONTRIBUTING.md).
