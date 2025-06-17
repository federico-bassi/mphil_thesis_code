"""
Creates a sequence of budgets constraints such that each passes through the
average‑optimal bundle of the previous budget (Optimal‑Placement
heuristic) and visualises them.
"""
import numpy as np
import pandas as pd
from maximize_utility_bc import *
import math
from fractions import Fraction
from data_generation import *

# ---------------------------------------------------------------------------
# Subject population generator (shared parameters across budgets)
# ---------------------------------------------------------------------------

def _generate_subjects(n_draws, param_dist, utility_fn):
    subjects = []
    for i in range(n_draws):
        θ = {k: draw() for k, draw in param_dist.items()}

        def u(x, y, _θ=θ):
            return utility_fn(x, y, **_θ)

        subjects.append((i, u, θ))
    return subjects


# ---------------------------------------------------------------------------
# Single‑budget optimiser (fixed subject pool)
# ---------------------------------------------------------------------------

def _optima_on_budget(budget, subjects, maximiser="precise"):
    ms, mo = budget
    rows = [(i, ms, mo, u, θ) for (i, u, θ) in subjects]
    df = pd.DataFrame(rows, columns=["id", "max_self", "max_other", "utility_func", "utility_label"])

    if maximiser == "fast":
        opt = maximise_utility(df)
    elif maximiser == "precise":
        opt = maximise_budget_precise(df)
    elif maximiser == "exact":
        opt = maximise_ces_exact(df)
    else:
        raise ValueError("Maximiser must be 'fast', 'precise', or 'exact'")

    return opt[["id", "opt_x", "opt_y"]]


def representative_optimum(budget, utility_fn, param_dist,
                           n_draws=100, maximiser="precise",
                           statistic="average"):
    """
    Return a representative bundle on a budget line.

    budget      tuple(max_self, max_other)
    utility_fn  u(x, y, **θ)
    param_dist  {'param': draw_function()}
    n_draws     how many parameter draws
    maximiser   'fast' | 'precise' | 'exact'
    statistic   'average' | 'median'
    """
    ms, mo = budget
    rows = []

    for i in range(n_draws):
        θ = {k: draw() for k, draw in param_dist.items()}

        def u_fixed(x, y, _θ=θ):
            return utility_fn(x, y, **_θ)

        rows.append((i, ms, mo, u_fixed, θ))

    df = pd.DataFrame(rows,
                      columns=["id", "max_self", "max_other",
                               "utility_func", "utility_label"])

    if maximiser == "fast":
        opt = maximise_utility(df)
    elif maximiser == "precise":
        opt = maximise_budget_precise(df)
    elif maximiser == "exact":
        opt = maximise_ces_exact(df)
    else:
        raise ValueError("maximiser must be 'fast', 'precise', or 'exact'")

    if statistic == "average":
        # coordinate‑wise mean (always lies on the budget)
        return opt[["opt_x", "opt_y"]].mean().to_numpy()

    elif statistic == "median":
        # take the median x and project it onto the budget line
        x_med = opt["opt_x"].median()
        y_proj = mo * (1 - x_med / ms)
        return np.array([x_med, y_proj])

    else:
        raise ValueError("statistic must be 'average' or 'median'")


def place_budgets(template, utility_fn, param_sampler, n_draws=100, optimiser="precise", statistic="average"):
    """ Given a list of prices in decreasing order, this algorithm shifts the income to maximize the optimal-placement
    heuristics.

    * template – list of (max_self, max_other) sorted by decreasing slope
    Returns (budgets_positioned, opt_points)
    """
    positioned = []
    opt_points = []

    # first budget as given
    first = template[0]
    positioned.append(first)
    opt = representative_optimum(first, utility_fn, param_sampler, n_draws, optimiser, statistic)
    opt_points.append(opt)

    prev_pt = opt
    for ms_t, mo_t in template[1:]:
        r = mo_t / ms_t
        x0, y0 = prev_pt
        ms_new = x0 + y0 / r
        mo_new = r * ms_new
        positioned.append((ms_new, mo_new))
        opt = representative_optimum((ms_new, mo_new), utility_fn, param_sampler, n_draws, optimiser, statistic)
        opt_points.append(opt)
        prev_pt = opt

    return positioned, opt_points


def place_budgets_garp_with_noise(
        template,
        utility_fn,
        param_sampler,
        *,
        n_draws: int = 100,
        maximiser: str = "exact",
        noise_std: float = 5.0,
):
    """
    Position a sequence of budget lines so that each new budget—chosen from the
    set of *integer-intercept* budgets that pivot the template line between the
    two adjacent corners—maximises the number of GARP violations it induces
    once semicircular noise is added to subjects’ optimal bundles.

    We search for budget constraints that intercept the two axes in integer values.

    Parameters
    ----------
    template : list[tuple[int, int]]
        Sequence of (max_self, max_other) pairs that define the *shape*
        (slope) of each budget to be placed.  Their absolute scale is ignored;
        only slopes matter.
    utility_fn, param_sampler
        Passed straight through to `_generate_subjects`.
    n_draws : int
        Number of subjects to simulate.
    maximiser : {"exact", …}
        Passed to `_optima_on_budget`.
    noise_std : float
        Standard deviation of the semicircular noise.

    Returns
    -------
    positioned : list[tuple[int, int]]
        The integer-intercept budgets that were actually placed.
    optima : list[pd.DataFrame]
        Optimal bundles (one DataFrame per budget).
    noisy_optima : list[pd.DataFrame]
        Noisy bundles (one DataFrame per budget).
    """
    # ---------- 0. sanity checks & first budget ------------------------------
    if not template:
        raise ValueError("template cannot be empty")

    subjects = _generate_subjects(n_draws, param_sampler, utility_fn)

    # Take the first budget exactly as given (income + price)
    positioned = [template[0]]
    # Compute the optima on the first budget
    opt_first = _optima_on_budget(template[0], subjects, maximiser)
    # Add the columns representing the budget constraint
    opt_first["max_self"], opt_first["max_other"] = template[0]
    # Add noise
    noisy_first = add_noise_semicircle(opt_first, std=noise_std, free_disposal=False)

    # Store optima, noisy optima and slope
    optima = [opt_first]
    noisy_optima = [noisy_first]
    slopes = [template[0][1] / template[0][0]]

    # ---------- 1. iterate over remaining template budgets -------------------
    for (ms_t, mo_t) in template[1:]:
        slope_curr = mo_t / ms_t
        if any(np.isclose(slope_curr, s) for s in slopes):
            raise ValueError("Each slope in *template* must differ from all earlier slopes.")

        # previous budget’s intercepts
        ms_prev, mo_prev = positioned[-1]

        # Take the greatest common divisor g
        g = math.gcd(ms_t, mo_t)
        # Peel off the common factor to obtain s1 and s2. We will use s1 and s2 to compute budgets that: (1) preserve
        # the original slope, (2) intercept the axes in "integers" values
        s1, s2 = ms_t // g, mo_t // g

        # Compute the smallest and largest scale factor allowed
        k_min, k_max = ms_prev / ms_t, mo_prev / mo_t

        # Compute the minimum and maximum income that are integer value and intercept the old budget constraint in
        # the leftmost allowed point and rightmost allowed point
        m_min = math.ceil(k_min * g)
        m_max = math.floor(k_max * g)
        if m_min > m_max:
            raise RuntimeError("No integer-intercept budget fits between the two corners.")

        # Prepare for iteration
        best_m, best_cnt = None, -1
        best_opt, best_noisy = None, None

        # ---------- 1a. search over integer-intercept candidates -------------
        for m in range(m_min, m_max + 1):
            ms_c, mo_c = m * s1, m * s2
            slope_c = slope_curr  # identical for all m

            # optimal and noisy bundles on the candidate budget
            opt_c = _optima_on_budget((ms_c, mo_c), subjects, maximiser)
            opt_c["max_self"], opt_c["max_other"] = ms_c, mo_c
            noisy_c = add_noise_semicircle(opt_c, std=noise_std, free_disposal=True)

            # ---- count unique subjects violating GARP wrt previous budgets
            violators = set()
            for j, (ms_j, mo_j) in enumerate(positioned):
                slope_j = mo_j / ms_j
                denom = slope_j - slope_c
                # (numerically) parallel lines
                if np.isclose(denom, 0):
                    continue

                # intersection on x-axis
                x_A = (mo_j - mo_c) / denom
                # intersection outside overlapping range
                if not (0 < x_A < min(ms_j, ms_c)):
                    continue

                noisy_prev = noisy_optima[j]
                merged = noisy_prev.merge(noisy_c, on="id", suffixes=("_p", "_c"))
                ids = merged.loc[
                    (merged.noisy_x_p > x_A) & (merged.noisy_x_c < x_A),
                    "id"
                ]
                violators.update(ids)

            if len(violators) > best_cnt:
                best_m, best_cnt = m, len(violators)
                best_opt, best_noisy = opt_c, noisy_c

        if best_m is None:
            raise RuntimeError("Unexpected: search loop produced no candidate.")

        # ---------- 1b. commit the best integer-intercept budget -------------
        ms_ch, mo_ch = best_m * s1, best_m * s2
        positioned.append((ms_ch, mo_ch))
        optima.append(best_opt)
        noisy_optima.append(best_noisy)
        slopes.append(slope_curr)

    return positioned, optima, noisy_optima


def find_integer_upward_slopes(budgets, max_n=10):
    """
    budgets : list[tuple[int, int]]
        (max_self, max_other) pairs – the x- and y-intercepts of downward
        budget lines.
    max_n   : int
        Generates the “nice” slope set
            1/max_n, …, 1/1, 1, 2, …, max_n   (default 10).

    Returns
    -------
    list[tuple[Fraction, tuple[int, int]]]
        Each item is ( slope , (x, y) )
          • slope – exact value as Fraction
          • (x, y) – integer lattice point where that upward line meets one of
            the downward lines.  Every valid pair appears once.
    """
    results = []          # stores (slope , (x , y))
    seen    = set()       # avoids duplicates

    # 1. build the list of “nice” slopes
    slopes = [Fraction(1, k) for k in range(max_n, 0, -1)]   # 1/max_n … 1
    slopes += [Fraction(k, 1) for k in range(1, max_n + 1)]  # 1 … max_n

    # 2. test every slope against every downward budget
    for m in slopes:                        # m is a Fraction
        for ms, mo in budgets:              # ms = max_self, mo = max_other
            # exact intersection: x = mo / ( m + (mo/ms) )
            denom = m + Fraction(mo, ms)
            x = Fraction(mo, 1) / denom

            # keep only if x is a positive integer
            if x.denominator != 1 or x <= 0:
                continue

            y = m * x                       # y on the upward line
            if y.denominator != 1 or y <= 0:
                continue                    # y must be integer too

            key = (m, x, y)                 # uniqueness signature
            if key not in seen:
                seen.add(key)
                results.append((m, (x.numerator, y.numerator)))

    # optional but handy: sort first by slope, then by x-coordinate
    results.sort(key=lambda t: (t[0], t[1][0]))
    return results

# ---------------------------------------------------------------------
# Simulate num_sim subjects per budget with random CES parameters
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Simulate num_sim subjects per budget with random parameters
# ---------------------------------------------------------------------
def budget_intersection_quantile(
        budgets,
        num_sim: int = 1,
        *,
        utility_func,
        param_distribution: dict[str, callable] | None = None,
        maximiser: str = "precise"
):
    """
    Simulate *num_sim* subjects on each budget, draw a fresh parameter
    set for every subject from *param_distribution*, optimise, locate
    the 45°-intersection, and report its quantile.
    """
    if param_distribution is None:
        param_distribution = {}

    # -------- build request --------------------------------------------------
    rows = []
    next_id = 0
    for b_idx, (ms, mo) in enumerate(budgets):
        for s_idx in range(num_sim):
            θ = {k: draw() for k, draw in param_distribution.items()}

            def u_sub(x, y, _f=utility_func, _θ=θ):
                return _f(x, y, **_θ)

            rows.append(dict(id=next_id,
                             budget_id=b_idx,
                             sim_id=s_idx,
                             max_self=ms,
                             max_other=mo,
                             utility_func=u_sub,
                             utility_label=θ))
            next_id += 1

    df_req   = pd.DataFrame(rows)
    meta_df  = df_req[["id", "budget_id", "sim_id"]]        # keep indices

    # -------- optimise -------------------------------------------------------
    if maximiser == "fast":
        opt = maximise_utility(df_req)
    elif maximiser == "precise":
        opt = maximise_budget_precise(df_req)
    elif maximiser == "exact":
        opt = maximise_ces_exact(df_req)
    else:
        raise ValueError("maximiser must be 'fast', 'precise', or 'exact'")

    # -------- bring indices back --------------------------------------------
    opt = opt.merge(meta_df, on="id", how="left")

    # -------- 45° intersection & quantile -----------------------------------
    opt["x_45"] = (opt["max_self"] * opt["max_other"]) / (
                    opt["max_self"] + opt["max_other"])

    xs = opt["opt_x"].to_numpy()
    opt["q_45"] = opt["x_45"].apply(lambda x: (xs < x).sum() / len(xs))

    # -------- unpack parameter dict to tidy columns -------------------------
    θ_cols = (pd.json_normalize(opt["utility_label"])
                .add_prefix("drawn_"))
    opt = pd.concat([opt.drop(columns=["utility_label"]), θ_cols], axis=1)

    # -------- final column order --------------------------------------------
    return opt[["id", "budget_id", "sim_id",
                "max_self", "max_other"] +
               list(θ_cols.columns) +
               ["opt_x", "opt_y", "x_45", "q_45"]]




__all__ = ["place_budgets", "place_budgets_garp_with_noise", "find_integer_upward_slopes",
           "budget_intersection_quantile"]
