import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import revpref as rp


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _generate_arrays(data):
    """Return price and quantity arrays for *revpref*.

    Prices are normalised so that the price of *x_self* is 1, and the
    price of *x_other* becomes *max_self / max_other*.
    """
    p = np.column_stack((np.ones(len(data)), data["max_self"] / data["max_other"]))
    q = np.column_stack((data["noisy_x"], data["noisy_y"]))
    return p, q


# ---------------------------------------------------------------------
# Index calculation
# ---------------------------------------------------------------------

def compute_index(data, index_type="CCEI"):
    """Compute the chosen consistency index for each (id, noise) pair.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain *id*, *noise*, *max_self*, *max_other*, *noisy_x*,
        *noisy_y*.
    index_type : str, optional
        Either "CCEI" (default) or "HMI".
    """
    results = []

    for (sid, noise), grp in data.groupby(["id", "noise"], observed=True):
        p, q = _generate_arrays(grp)
        pref = rp.RevealedPreference(p, q)

        try:
            if index_type.upper() == "HMI":
                value = pref.hmi()
            else:
                value = pref.ccei()
        except Exception:
            value = np.nan

        results.append({"id": sid, "noise": noise, index_type: value})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Binning utilities
# ---------------------------------------------------------------------

def make_bin_labels(edges):
    labels = [f"[{edges[i]:.2f}, {edges[i + 1]:.2f})" for i in range(len(edges) - 2)]
    labels.append("1")
    return labels


def bin_index(
        data: pd.DataFrame,
        *,
        index_type: str = "CCEI",
        group_by: str = "noise",
        bin_width: float = 0.05
) -> pd.DataFrame:
    """
    Bin an index (e.g. CCEI, HMI) and compute the share of observations
    in each bin *within* every level of *group_by*.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain the `index_type` column plus the column named
        in *group_by* (e.g. 'noise' or 'pucktreatment').
    index_type : str, default "CCEI"
        Which consistency index column to bin.
    group_by : str, default "noise"
        Variable that defines the hue in the downstream plot.
    bin_width : float, default 0.05
        Width of the bins on the index domain [0, 1].

    Returns
    -------
    pd.DataFrame with columns ["bin", group_by, "fraction"].
    """

    # ---------------- create bins ----------------
    edges  = np.arange(0, 1.0 + bin_width + 1e-9, bin_width)
    labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
              for i in range(len(edges) - 2)] + ["1"]

    df = data.copy()
    df["bin"] = pd.cut(
        df[index_type],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False
    )
    df["bin"] = (
        pd.Categorical(df["bin"], categories=labels, ordered=True)
          .remove_unused_categories()
    )

    # ------------- tidy the grouping variable -------------
    if group_by not in df.columns:
        raise KeyError(f"'{group_by}' not found in the supplied DataFrame.")

    if group_by == "noise":
        cats = sorted(df[group_by].unique(), reverse=True)  # σ=0 on right
    else:
        cats = sorted(df[group_by].unique())

    df[group_by] = pd.Categorical(df[group_by], categories=cats, ordered=True)

    # ---------------- aggregate & normalise ---------------
    binned = (
        df.groupby(["bin", group_by], observed=True)
          .size()
          .reset_index(name="count")
    )
    totals = df.groupby(group_by, observed=True).size()
    binned["fraction"] = binned.apply(
        lambda r: r["count"] / totals.loc[r[group_by]], axis=1
    )

    return binned



# ---------------------------------------------------------------------
# GARP‑violation analysis
# ---------------------------------------------------------------------

def count_violations(data):
    """Return a DataFrame that counts how often each budget triggers GARP violations."""
    counts = {}

    for (sid, noise), grp in data.groupby(["id", "noise"], observed=True):
        unique_budgets = grp[["max_self", "max_other"]].drop_duplicates()

        for _, row in unique_budgets.iterrows():
            max_self = row["max_self"]
            max_other = row["max_other"]

            subset = grp[~((
                                   grp["max_self"] == max_self) & (grp["max_other"] == max_other))]

            # Cannot check GARP with fewer than two choices
            if len(subset) < 2:
                continue

            p_full, q_full = _generate_arrays(grp)
            p_sub, q_sub = _generate_arrays(subset)

            pref_full = rp.RevealedPreference(p_full, q_full)
            pref_sub = rp.RevealedPreference(p_sub, q_sub)

            if pref_sub.check_garp() and not pref_full.check_garp():
                key = (noise, max_self, max_other)
                counts[key] = counts.get(key, 0) + 1

    rows = [(*k, v) for k, v in counts.items()]
    return pd.DataFrame(rows, columns=[
        "noise", "max_self", "max_other", "garp_trigger_count"])
