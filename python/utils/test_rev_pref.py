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

def compute_index(data, index_type="CCEI", simple_check = False):
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

        if simple_check:
            value = pref.check_garp()
            index_type = "GARP"
        else:
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

# ---------------------------------------------------------------------
#   UTILITY-LOSS ANALYSIS
# ---------------------------------------------------------------------
# Computes the "expenditure" index
def _ces_unit_expenditure(p, alpha, rho):
    p = np.asarray(p, float)
    alpha = np.asarray(alpha, float)
    rho = np.asarray(rho, float)

    sigma = 1.0/(1.0- rho)
    term = np.power(alpha, sigma) + np.power(1.0 - alpha, sigma) * np.power(p, 1.0 - sigma)
    return np.power(term, 1.0 / (1.0 - sigma))

# Extract alpha and rho from the dictionary "utility_label", returns two aligned arrays
def _extract_alpha_rho_from_label(series):
    a = series.apply(lambda d: d["alpha_ces"])
    r = series.apply(lambda d: d["rho_ces"])
    return a.to_numpy(dtype=float), r.to_numpy(dtype=float)

# Takes the dataframe and computes the money-metric loss for each row, stored in two columns
def attach_money_metric_loss(
        df: pd.DataFrame,
        ces_func,
        *,
        s_col = "noisy_x",
        o_col = "noisy_y",
        loss_col = "welfare_loss",
        loss_rate_col = "welfare_loss_rate"
):
    d = df.copy()

    # Compute the relative income and relative price
    m = d["max_self"].to_numpy(dtype=float)
    p = (d["max_self"] / d["max_other"]).to_numpy(dtype=float)

    # Extract the individual-level parameters from the column "utility_label"
    alpha, rho = _extract_alpha_rho_from_label(d["utility_label"])

    # Extract the chosen (noisy) point from the relevant columns
    s = d[s_col].to_numpy(dtype=float)
    o = d[o_col].to_numpy(dtype=float)

    # Compute the utility of the chosen bundle
    u = np.fromiter(
        (ces_func(si, oi, ai, ri) for si, oi, ai, ri in zip(s, o, alpha, rho)),
        dtype=float,
        count=len(d)
    )

    # Compute the monetary loss
    C = _ces_unit_expenditure(p, alpha, rho)
    L = m - u * C

    d[loss_col] = L
    d[loss_rate_col] = L / m
    return d


def summarize_welfare_loss_by_noise(df_with_loss, noise_col="noise", id_col="id", cons_key_cols=None,
                                    loss_col="welfare_loss", loss_rate_col="welfare_loss_rate"):
    """
    Summarizes welfare loss by noise level, computing the *average individual loss across all constraints*.

    Steps:
    1. Aggregate losses at the individual level by summing across all constraints.
    2. Average these individual totals for each noise level.
    """
    if cons_key_cols is None:
        # If constraints are identified by something other than noise/id, you can specify here
        cons_key_cols = []

    # 1. Aggregate to individual level across all constraints
    id_level = (
        df_with_loss
        .groupby([id_col, noise_col], observed=True)[[loss_col, loss_rate_col]]
        .sum()
        .reset_index()
    )

    # 2. Average across individuals within each noise level
    loss_summary = (
        id_level
        .groupby(noise_col, observed=True)[loss_col]
        .mean()
        .reset_index(name=f"mean_{loss_col}")
    )

    rate_summary = (
        id_level
        .groupby(noise_col, observed=True)[loss_rate_col]
        .mean()
        .reset_index(name=f"mean_{loss_rate_col}")
    )

    # 3. Merge into a single DataFrame
    summary_df = pd.merge(loss_summary, rate_summary, on=noise_col)

    return summary_df