"""Reference metric comparisons; scientific acceptance is a separate gate."""

import numpy as np
import pandas as pd


TRIALS = ("baseline", "infonce", "mmd", "ot")
METRICS = {
    "fixed_triplet_margin_loss": (0.02, "max"),
    "label_same_neighbor_fraction": (0.05, "min"),
    "species_mixing_fraction": (0.03, "min"),
}


def compare_metrics(reference, candidate):
    """Apply the defended evaluator's degradation budgets to each reference trial.

    Improvements are allowed. A candidate's own benchmark acceptance must still
    be checked separately; agreement with a failing reference is insufficient.
    """
    tables = []
    for name, frame in (("reference", reference), ("candidate", candidate)):
        required = {"trial", "fixed_triplet_count", *METRICS}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} comparison is missing required metrics")
        if len(frame) != len(TRIALS) or set(frame["trial"]) != set(TRIALS):
            raise ValueError(f"{name} must contain exactly the four defended trials")
        table = frame.set_index("trial").loc[list(TRIALS)]
        values = table[list(METRICS) + ["fixed_triplet_count"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} metrics must be finite and nonnegative")
        if (values[:, 1:3] > 1).any():
            raise ValueError(f"{name} neighborhood fractions must be at most one")
        counts = table["fixed_triplet_count"].to_numpy(dtype=float)
        if (counts <= 0).any() or not np.equal(counts, np.floor(counts)).all() or len(set(counts)) != 1:
            raise ValueError(f"{name} must use one nonempty frozen triplet set")
        tables.append(table)
    reference, candidate = tables
    if not np.array_equal(reference["fixed_triplet_count"], candidate["fixed_triplet_count"]):
        raise ValueError("Reference and candidate triplet counts differ")
    rows = []
    for trial in TRIALS:
        for metric, (tolerance, direction) in METRICS.items():
            expected, actual = float(reference.loc[trial, metric]), float(candidate.loc[trial, metric])
            degradation = actual - expected if direction == "max" else expected - actual
            rows.append({"trial": trial, "metric": metric, "reference": expected,
                         "candidate": actual, "difference": actual - expected,
                         "allowed_degradation": tolerance, "passes": degradation <= tolerance + 1e-12})
    return pd.DataFrame(rows)
