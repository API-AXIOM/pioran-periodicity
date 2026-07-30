"""pioran_periodicity: Bayesian periodicity detection in AGN light curves.

GP noise models (DRW, CARMA, one-bend power law via Pioran.jl) compared
against periodic alternatives with nested sampling (ultranest).

Design rules (see the accompanying fix report):
* shared parameters have identical priors in null and alternative models;
* process variances are sampled parameters -- models never normalise
  themselves against the data being fitted;
* PSD approximation bands are explicit, robust to sampling-pattern
  outliers, and their accuracy is measurable;
* all log parameters are base-10; every sampler setting is recorded.
"""

from .visualization import (
    filter_table,
    load_summary,
    plot_detection_rate,
    plot_fpr_calibration,
    plot_matched_roc,
    plot_power_curves,
    plot_roc,
    plot_strip,
    roc_from_bf,
)

# .visualization must import matplotlib.pyplot before anything below pulls in
# pioranpy: juliacall's Julia init prepends its own libexpat to
# DYLD_LIBRARY_PATH, and matplotlib (via font_manager -> plistlib) needs the
# system libexpat -- whichever loads first wins the process-wide symbol
# table, so pyplot must be imported first.
from .data import load_pg1302, load_pg1553, load_photometry_csv, load_three_column
from .inference import (
    FitResult,
    SamplerSettings,
    fit_family,
    load_result,
    log10_bayes_factors,
    run_nested,
    save_result,
)
from .kernels import (
    FrequencyBand,
    carma_kernel,
    drw_kernel,
    gp_log_likelihood,
    obpl_kernel,
    psd_approximation_error,
)
from .means import combine_means, linear_mean, sine_amplitude_phase, sine_mean
from .models import ModelFamily, ModelSpec, PriorConfig, build_family
from .priors import (
    ConditionalUniform,
    LogUniform,
    Normal,
    Parameter,
    PriorTransform,
    Uniform,
)

__version__ = "0.1.3"

__all__ = [
    "Uniform",
    "Normal",
    "LogUniform",
    "ConditionalUniform",
    "Parameter",
    "PriorTransform",
    "sine_mean",
    "linear_mean",
    "sine_amplitude_phase",
    "combine_means",
    "FrequencyBand",
    "drw_kernel",
    "obpl_kernel",
    "carma_kernel",
    "gp_log_likelihood",
    "psd_approximation_error",
    "PriorConfig",
    "ModelSpec",
    "ModelFamily",
    "build_family",
    "SamplerSettings",
    "FitResult",
    "run_nested",
    "fit_family",
    "log10_bayes_factors",
    "save_result",
    "load_result",
    "load_pg1302",
    "load_pg1553",
    "load_photometry_csv",
    "load_three_column",
    "load_summary",
    "filter_table",
    "plot_strip",
    "plot_power_curves",
    "plot_detection_rate",
    "plot_fpr_calibration",
    "plot_roc",
    "plot_matched_roc",
    "roc_from_bf",
]
