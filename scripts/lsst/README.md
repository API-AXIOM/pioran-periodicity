# LSST scripts

Survey-specific drivers for LSST work that the core package does not cover:
turning an OpSim visit history into a usable footprint/density map, and
re-expressing a campaign result as a population rate.

These carry extra dependencies the rest of `scripts/` does not need:

```
pip install healpy astroquery     # scipy and astropy come with the package
```

## Why these exist

The LSST target lists were drawn from Milliquas with a **declination-
stratified** sample (`build_target_lists.py`, `DEC_BIN_WIDTH = 15`), and the
catalogue was never cached. A campaign built on that list does not, on its
face, estimate a rate for any real population: its mix of densely and sparsely
sampled objects is a sampling choice. These scripts rebuild the parent and
reweight onto it.

Note it is the *target list* that is stratified, not the cadence library —
the library is a faithful subsample of the list.

## Pipeline

```
build_visit_map.py   OpSim .db            -> visit_counts_nside64.npy
fetch_parent.py      VizieR + visit map   -> parent_lsst.csv
reweight_fpr.py      campaign + parent    -> post-stratified FPR per cell
```

### `build_visit_map.py`

HEALPix visit-count map (nside 64, 1.75 deg FOV). Counts per pixel with a
KD-tree instead of per visit with `query_disc`; same map, far faster. Pass
`--validate-targets` with an archived target list — every object in it must
fall inside the resulting footprint, or the map does not reproduce the
selection the campaign was built on and nothing downstream is meaningful.

### `fetch_parent.py`

Re-queries Milliquas and applies the original cuts in the original order:
type-Q, `|b| > 15`, `16 <= R <= 23`, then `>= 100` visits. The VizieR query
pulls ~1e6 rows and takes a few minutes.

### `reweight_fpr.py`

`FPR_pop = sum_h W_h p_h`. Strata are cut on **visit count**, not declination:
sampling density is the causal driver, declination adds nothing once density
is in the model, and visit count is the only density measure available for
every parent object. Repeat `--edges` to check the answer against several
stratifications — a value that moves between them is riding on strata with
few campaign objects, not on a real difference.

Both gates matter. Pairs are dropped when either fit reports
`converged: false` *and* when either has an ESS below `--ess-min`. A nested
sampling run can report `converged: true` with `ess = 1` and a `logz` wrong by
sixteen orders of magnitude; only the second gate catches it.

## Interpreting the output

The reweighted rate is a rate for **Milliquas quasars inside the OpSim
footprint**, which is an observational compilation, not the true quasar
population. The default `--min-visits 100` footprint is also permissive: it
admits the survey's marginal northern edge, where objects are sampled far more
sparsely than in the main survey. "All footprint objects" and "WFD-like
objects" are different populations and give different answers; report which
one a number refers to.
