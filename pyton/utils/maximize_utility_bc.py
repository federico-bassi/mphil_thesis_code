import ast
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize

# UTILITY MAXIMIZATION # -----------------------------------------------------------------------------------------------
def maximise_utility(df):
    """Optimizes utility functions subject to budget constraints.

    Parameters:
    df (pd.DataFrame): DataFrame containing budget constraints and utility functions.

    Returns:
    pd.DataFrame: DataFrame containing optimization results.
    """
    results = []

    # Iterate over each rows
    for _, row in df.iterrows():
        # Define the negative utility for minimization
        def neg_utility(z):
            return -row['utility_func'](z[0], z[1])

        # Budget bounds
        bounds = [(0, row['max_self']), (0, row['max_other'])]

        def budget_constraint(max_self, max_other, x):
            return max_other - (max_other / max_self) * x

        # Budget constraint: y ≤ (max_other / max_self) * x
        cons = ({'type': 'ineq', 'fun': lambda z: budget_constraint(row['max_self'], row['max_other'], z[0]) - z[1]})

        # Use an interior point as starting value to avoid boundary issues
        initial_guess = np.array([15, 15])

        # Run constrained optimization with reasonable precision
        result = minimize(
            neg_utility,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "ftol": 1e-6,
                "eps": 1e-11,
                'maxiter': 100000,
                'disp': False
            }
        )

        if result.success:
            results.append((
                row["id"],
                row['max_self'],
                row['max_other'],
                float(result.x[0]),
                float(result.x[1]),
                row['utility_label']
            ))
        else:
            print(f"[WARNING] Optimization failed for ID {row['id']} with message: {result.message}")

    return pd.DataFrame(results, columns=["id", 'max_self', 'max_other', 'opt_x', 'opt_y', 'utility_label'])


def maximise_budget_precise(df):
    out = []
    for _, row in df.iterrows():
        ms, mo = row["max_self"], row["max_other"]
        u = row["utility_func"]

        # 1‑D utility in t
        f = lambda t: -u(t * ms, (1 - t) * mo)

        sol = minimize_scalar(f,
                              bounds=(0., 1.),
                              method="bounded",
                              options={"xatol": 1e-12,
                                       "maxiter": 50_000})

        # explicit corners
        cand = [(sol.x, sol.fun),
                (0.0, f(0.0)),
                (1.0, f(1.0))]
        t_best, _ = min(cand, key=lambda p: p[1])

        out.append((row["id"], ms, mo,
                    float(t_best * ms), float((1 - t_best) * mo),
                    row["utility_label"]))
    return pd.DataFrame(out,
                        columns=["id", "max_self", "max_other",
                                 "opt_x", "opt_y", "utility_label"])


def maximise_ces_exact(df):
    """
    For every row in *df* compute the CES-optimal payoffs (opt_x, opt_y)
    on a linear budget defined by the intercepts max_self (ms) and max_other (mo).

    Assumes each row has:
        max_self, max_other,
        utility_label = {"alpha_ces": a, "rho_ces": rho}
    """
    rows = []

    for _, row in df.iterrows():
        ms, mo = row["max_self"], row["max_other"]
        a = row["utility_label"]["alpha_ces"]
        rho = row["utility_label"]["rho_ces"]

        # Comput the relative price p  (= P_o / P_s) and the total "income" m'. Assume P_s is the numeraire.
        p = ms / mo
        m_ = ms

        # Elasticity of substitution
        sigma = 1.0 / (1.0 - rho)

        # Demand for own-payoff
        num = a ** sigma
        den = num + (1 - a) ** sigma * p ** (1 - sigma)
        opt_x = m_ * num / den

        # Corresponding other-payoff
        opt_y = (m_ - opt_x) / p

        rows.append((row["id"], ms, mo, opt_x, opt_y, row["utility_label"]))

    return pd.DataFrame(
        rows,
        columns=["id", "max_self", "max_other", "opt_x", "opt_y", "utility_label"]
    )


__all__ = [
    "maximise_utility",
    "maximise_budget_precise",
    "maximise_ces_exact",
]
