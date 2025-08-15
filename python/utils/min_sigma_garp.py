
from __future__ import annotations
from typing import Iterable, Literal, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Import from your existing modules
from data_generation import simulate_dataset, ces
from test_rev_pref import compute_index


def _boolean_has_violation(df: pd.DataFrame) -> Tuple[bool, int, float]:
    """
    Return (has_violation, n_violators, violator_share) for a dataframe that
    contains choices for many (id, noise) pairs.

    We call compute_index(..., simple_check=True), which returns a DataFrame
    with a boolean column 'GARP' indicating whether GARP is satisfied for
    each (id, noise) pair. A violation corresponds to GARP == False.
    """
    idx = compute_index(df, index_type="GARP", simple_check=True)  # columns: id, noise, GARP (bool)
    # If someone violated, GARP is False
    violators = (~idx["GARP"].astype(bool)).sum()
    total     = len(idx)
    share     = (violators / total) if total else 0.0
    return violators > 0, int(violators), float(share)


def _as_grid(start: float, stop: float, step: float) -> np.ndarray:
    """
    Numerically stable [start, stop] grid with decimal steps.
    Ensures inclusive of stop (to 1e-9 tolerance).
    """
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    return np.round(start + step * np.arange(n), 10)


def min_sigma_for_garp(
    *,
    budgets: Iterable[Tuple[float, float]],
    utility_func,
    param_distributions: Dict[str, Any],
    n_samples: int = 50,
    sigma_start: float = 0.1,
    sigma_stop: float = 20.0,
    sigma_step: float = 0.1,
    disposal: Literal["free", "nodis"] = "free",
    maximiser: Literal["fast", "precise", "exact"] = "fast",
    seed: Optional[int] = None,
    mc_reps: int = 1,
) -> Dict[str, Any]:
    """
    Scan sigma in [sigma_start, sigma_stop] with step sigma_step. For each sigma,
    simulate a dataset and check for any GARP violation. Stop at first sigma
    that yields at least one violation.

    Parameters
    ----------
    budgets : iterable of (max_self, max_other)
    utility_func : callable u(x, y, **params)
    param_distributions : dict of {param_name: zero-arg sampler}
    n_samples : number of individuals to simulate
    sigma_start, sigma_stop, sigma_step : grid definition (inclusive stop if hit exactly)
    disposal : 'free'  → uses free-disposal noisy data (moves inside the triangle)
               'nodis' → uses no-disposal noisy data (moves along the budget line)
    maximiser : 'fast' | 'precise' | 'exact' (passed to simulate_dataset)
    seed : base RNG seed. If provided, each (sigma, rep) uses seed + rep for reproducibility.
    mc_reps : Monte Carlo repetitions per sigma; stop if any rep violates.

    Returns
    -------
    dict with keys:
        - 'sigma_min'       : first sigma with a violation (None if none found)
        - 'violators'       : number of violators at that sigma/rep
        - 'share_violators' : share of violators at that sigma/rep
        - 'disposal'        : the disposal mode used
        - 'mc_rep'          : which rep triggered stopping (0-indexed)
        - 'checked_sigmas'  : list of sigma values that were tested
        - 'grid_params'     : (start, stop, step)
    """
    if sigma_step <= 0:
        raise ValueError("sigma_step must be positive.")
    if sigma_start <= 0:
        raise ValueError("sigma_start must be positive.")

    grid = _as_grid(sigma_start, sigma_stop, sigma_step)
    checked = []

    for s in grid:
        for r in range(mc_reps):
            # Make runs reproducible but varied across reps if a seed is provided
            run_seed = None if seed is None else (int(seed) + r)

            df_free, df_nodis = simulate_dataset(
                budgets=budgets,
                utility_func=utility_func,
                param_distributions=param_distributions,
                n_samples=n_samples,
                noise_sd=[s],
                maximiser=maximiser,
                seed=run_seed,
            )

            df = df_free if disposal == "free" else df_nodis

            has_violation, n_viol, share = _boolean_has_violation(df)
            checked.append(float(s))

            if has_violation:
                return {
                    "sigma_min": float(s),
                    "violators": int(n_viol),
                    "share_violators": float(share),
                    "disposal": disposal,
                    "mc_rep": r,
                    "checked_sigmas": checked,
                    "grid_params": (float(sigma_start), float(sigma_stop), float(sigma_step)),
                }

    # If we get here: no violation on the scanned grid
    return {
        "sigma_min": None,
        "violators": 0,
        "share_violators": 0.0,
        "disposal": disposal,
        "mc_rep": None,
        "checked_sigmas": checked,
        "grid_params": (float(sigma_start), float(sigma_stop), float(sigma_step)),
    }


# Optional convenience: a ready-to-use CES wrapper with uniform priors
def min_sigma_for_garp_ces(
    *,
    budgets: Iterable[Tuple[float, float]],
    alpha_low: float = 0.25,
    alpha_high: float = 0.9,
    rho_low: float = -10.0,
    rho_high: float = 0.5,
    n_samples: int = 100,
    sigma_start: float = 0.1,
    sigma_stop: float = 20.0,
    sigma_step: float = 0.1,
    disposal: Literal["free", "nodis"] = "free",
    maximiser: Literal["fast", "precise", "exact"] = "exact",
    seed: Optional[int] = 123,
    mc_reps: int = 1,
) -> Dict[str, Any]:
    """
    CES-specialised helper with uniform draws for alpha and rho.
    """
    param_distributions = {
        "alpha_ces": lambda: np.random.uniform(alpha_low, alpha_high),
        "rho_ces":   lambda: np.random.uniform(rho_low, rho_high),
    }
    return min_sigma_for_garp(
        budgets=budgets,
        utility_func=ces,
        param_distributions=param_distributions,
        n_samples=n_samples,
        sigma_start=sigma_start,
        sigma_stop=sigma_stop,
        sigma_step=sigma_step,
        disposal=disposal,
        maximiser=maximiser,
        seed=seed,
        mc_reps=mc_reps,
    )


if __name__ == "__main__":
    # --- Example usage (fill in your own budgets) ---
    # Andreoni–Miller budgets (example placeholder; replace with your list)
    andreoni_miller_budgets = [
        # (max_self, max_other), e.g.
        # (80, 80), (100, 60), (60, 100), ...
    ]

    # Quick smoke run (won't run meaningfully until budgets are provided)
    if andreoni_miller_budgets:
        out = min_sigma_for_garp_ces(
            budgets=andreoni_miller_budgets,
            n_samples=200,
            sigma_start=0.1,
            sigma_stop=20.0,
            sigma_step=0.1,
            disposal="free",
            maximiser="exact",
            seed=42,
            mc_reps=5,
        )
        print(out)
    else:
        print("Fill in 'andreoni_miller_budgets' with your constraints to run the example.")
