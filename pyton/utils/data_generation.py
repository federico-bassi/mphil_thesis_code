import pandas as pd
import numpy as np
from maximize_utility_bc import (
    maximise_utility,
    maximise_budget_precise,
    maximise_ces_exact,
)


# STANDARD UTILITY FUNCTIONS # -----------------------------------------------------------------------------------------
def fehr_schmidt_utility(x, y, alpha=0.5, beta=0.5):
    """Computes Fehr-Schmidt utility"""
    return x - (alpha * (y - x) if y > x else beta * (x - y))


def altruistic_utility(x, y, theta=0.5):
    """Computes Altruistic utility, where utility increases with other's payoff."""
    return x + theta * y


def selfish_utility(x, y):
    """Selfish utility function, considering only self-payoff."""
    return x


def cobb_douglas(x, y, alpha=0.5, beta=0.5):
    """Computes Cobb-Douglas utility, assuming a trade-off between self and other."""
    return x ** alpha * y ** beta


def ces(x, y, alpha_ces=0.5, rho_ces=0):
    eps = 1e-8  # a very small threshold to avoid numerical issues

    if x < eps or y < eps:
        return 0  # enforce 0 utility when near-zero inputs would break power ops

    if abs(rho_ces) < eps:
        return x ** alpha_ces * y ** (1 - alpha_ces)

    try:
        return (alpha_ces * x ** rho_ces + (1 - alpha_ces) * y ** rho_ces) ** (1 / rho_ces)
    except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
        print(f"Invalid CES input: x={x}, y={y}, alpha={alpha_ces}, rho={rho_ces} — Error: {e}")
        return 0


# BUDGET CONSTRAINTS OVER MAX_SELF, MAX_OTHER # ------------------------------------------------------------------------
def budget_constraint(max_self, max_other, x):
    """Defines the budget constraint given maximum resources."""
    return max_other - (max_other / max_self) * x


# DATA GENERATION # ----------------------------------------------------------------------------------------------------
def data_generation(budgets, utility_funcs):
    """
    Parameters:
    - budgets: list of (max_self, max_other)
    - utility_funcs: list of (func, param_dict, label)
      e.g. [(ces, {'alpha_ces': 0.5, 'rho_ces': -10}, 'CES_1'), ...]
    """
    data = []
    ind = 1
    for max_self, max_other in budgets:
        for func, param_dict, label in utility_funcs:
            def utility_fixed(x, y, func=func, param_dict=param_dict):
                return func(x, y, **param_dict)

            data.append((ind, max_self, max_other, utility_fixed, label))
            ind += 1
    return pd.DataFrame(data, columns=["id", "max_self", "max_other", "utility_func", "utility_label"])


def data_generation_random_parameters(
        budgets,
        utility_func,
        param_distributions,
        n_samples: int = 50
):
    """
    Generate a DataFrame of (id, budget, utility) pairs, where the `utility_label`
    column stores the exact parameter draw used for that row.

    Parameters
    ----------
    budgets : list[tuple[float, float]]
        List of budget constraints defined by (max_self, max_other).
        For example: andreoni_miller_budgets = [(120, 40), (40, 120), ...]

    utility_func : callable
        Base utility function of the form u(x, y, **params).

    param_distributions : dict[str, callable]
        Keys are parameter names; values are zero‑argument callables that return a random draw.
        For example: {'alpha_ces': lambda: np.random.uniform(0.5, 1), 'rho_ces': lambda: np.random.uniform(-10, 1)}

    n_samples : int, default 50
        Number of individuals in the dataset

    Returns
    -------
    pd.DataFrame
        Columns:
        - id              : id of each simulated subject
        - max_self        : max for oneself
        - max_other       : max fot other
        - utility_func    : concrete utility function with params fixed
        - utility_label   : **dict** of the sampled parameters
    """

    rows = []
    # For each individual in the sample, draw a value of parameters. Then, create a utility function with the sampled
    # parameters. For each individual and for each budget constraint, report the instatiated utility function in a
    # dataframe.
    for i in range(1, n_samples + 1):
        sampled_params = {k: draw() for k, draw in param_distributions.items()}

        def utility_fixed(x, y, _p=sampled_params):
            # **_p unpacks the dict into keyword arguments: e.g. utility_func(x, y, alpha_ces=0.63, rho_ces=-1.4)
            return utility_func(x, y, **_p)

        for max_self, max_other in budgets:
            rows.append((i, max_self, max_other,
                         utility_fixed, sampled_params))

    return pd.DataFrame(
        rows,
        columns=["id", "max_self", "max_other", "utility_func", "utility_label"]
    )


def add_noise_semicircle(df, std=5, free_disposal=True):
    """
    Add semicircular noise while **always** keeping the noisy point inside
    the triangle with vertices (0,0), (max_self,0), (0,max_other).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns
            'opt_x', 'opt_y',         # optimal bundle (on the budget line)
            'max_self', 'max_other'   # intercepts of the budget line
    std : float, default 5
        standard deviation of the half‑normal distribution for the step length.
    free_disposal : bool, default True
        • True  → rotation angle is Uniform (‑90°, +90°) around the inward unit vector.
        • False → rotation angle is either ‑90° or +90° (moves along the budget line).

    Returns
    -------
    pd.DataFrame
        Original frame plus
            'noisy_x', 'noisy_y' : the perturbed bundle,
            'noise'              : std (for reference).
    """
    df = df.copy()

    # Iterate over the rows
    for idx, row in df.iterrows():
        # ---------- 1. geometry parameters -------------------------
        p0 = np.array([row['opt_x'], row['opt_y']])
        ms, mo = row['max_self'], row['max_other']

        # prices (p₁ = 1, p₂ = ms/mo). Price of x_self is numeraire, the price for payoff to other is relative
        p2 = ms / mo

        # inward‑pointing unit vector to the budget line
        n_in = np.array([-1.0, -p2])
        n_in /= np.linalg.norm(n_in)

        # ---------- 2. choose a direction inside the triangle -------
        # If free_disposal is true, choose any direction between -90 and +90 wrt the tangent, otherwise only stick to
        # one of the two.
        if free_disposal:
            angle_deg = np.random.uniform(-90, 90)
        else:
            angle_deg = np.random.choice([-90, 90])

        # Express the angle is radiant
        theta = np.deg2rad(angle_deg)
        # Rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        # Compute the new direction
        d = R @ n_in

        # ---------- 3. prevent points from moving outside of feasible region -----------
        # Create an empty list of candidates steps
        candidates = []

        # Take the direction d that will be used to project the point. If the direction is pointing left (i.e. d[0] <
        # 0), compute the maximum allowable step towards the left, defined by the equation: p0[0] + t * d[0] = 0,
        # or equivalently t=-p0[0] / d[0]. If the direction is pointing down (i.e. d[1] < 0), compute max allowable
        # step downwards. Then, if the length drawn from the normal distribution is larger than the minimum between the
        # two lengths just computed, cap the drawn length at this minimum. In this way, points whose noisy projection
        # would have been located in a non-feasible region are restricted to stay in that region.
        if d[0] < 0:
            candidates.append(-p0[0] / d[0])
        if d[1] < 0:
            candidates.append(-p0[1] / d[1])

        # Compute the first boundary that the perturbation vector would hit
        t_max = min(candidates) if candidates else np.inf

        # ---------- 4. draw step length and clip -------------------
        length = abs(np.random.normal(scale=std))

        # If the length of perturbation is too large (larger than t_max), use t_max as a perturbation length
        length = min(length, t_max)

        # ---------- 5. compute the noisy point --------------------
        noisy_point = p0 + length * d
        df.loc[idx, ['noisy_x', 'noisy_y']] = np.maximum(noisy_point, 0.0)

    df['noise'] = std
    return df


def compute_distance(df):
    """
    Computes the Euclidean distance between noisy and optimal choices.

    Assumes the DataFrame contains:
    - 'opt_x', 'opt_y': optimal choices
    - 'noisy_x', 'noisy_y': noisy observed choices

    Returns:
        pd.DataFrame: with added 'distance' column.
    """
    distances = np.sqrt(
        (df["noisy_x"] - df["opt_x"]) ** 2 +
        (df["noisy_y"] - df["opt_y"]) ** 2
    )
    df["distance"] = distances

    print("Key distance metrics")
    for noise in df.noise.unique():
        if noise == 0:
            continue
        print("Noisy data with standard deviation of: ", noise)
        for i in range(1, 4):
            total = df.loc[(df.noise == noise), ["distance"]].shape[0]
            count = df.loc[(df.noise == noise) & (df.distance <= i * noise), ["distance"]].shape[0]
            print(round((count / total * 100), 1), " observations have a distance lower or equal to ", i * noise,
                  "from the optimum.")
        print("\n")
    return df


def simulate_dataset(
        budgets,  # list[(max_self, max_other)]
        utility_func,  # u(x, y, **params)
        param_distributions,  # dict {param: draw()}
        n_samples=50,  # number of simulations
        noise_sd=5.,  # pass a list or an exact value
        maximiser="fast",  # "fast" | "precise" | "exact"
):
    """
    Draw random parameters → maximise utility → add semicircle noise *twice*:
      • free-disposal  (moves only along the budget line)
      • no-disposal    (can also move inward)

    Returns
    -------
    (df_free, df_nodis)  tuple of two DataFrames
    """
    # 1 ─ parameter draws
    df_base = data_generation_random_parameters(
        budgets, utility_func, param_distributions, n_samples
    )

    # 2 ─ maximize
    if maximiser == "fast":
        df_opt = maximise_utility(df_base)
    elif maximiser == "precise":
        df_opt = maximise_budget_precise(df_base)
    elif maximiser == "exact":
        df_opt = maximise_ces_exact(df_base)
    else:
        raise ValueError("maximiser must be 'fast', 'precise', or 'exact'")

    # 3 ─ add noise under both disposal assumptions
    if np.iterable(noise_sd) and not isinstance(noise_sd, str):
        sds = noise_sd
    else:
        sds = [noise_sd]

    free_frames = [add_noise_semicircle(df_opt, std=sd, free_disposal=True) for sd in sds]
    nodis_frames = [add_noise_semicircle(df_opt, std=sd, free_disposal=False) for sd in sds]

    df_free = pd.concat(free_frames, ignore_index=True)
    df_nodis = pd.concat(nodis_frames, ignore_index=True)
    return df_free, df_nodis


def budgets_algorithm_ces(param_distribution, starting_budget_set=(60, 60), num_budgets=11, slope_sens=0.8,
                          digit_sens=2, num_simulations=200):
    budgets = [starting_budget_set]
    starting_x, starting_y = starting_budget_set
    for i in range(num_budgets):
        df, df2 = simulate_dataset(budgets=[starting_budget_set], utility_func=ces,
                                   param_distributions=param_distribution, n_samples=num_simulations,
                                   noise_sd=[0], maximiser="exact")
        counts = round(df.loc[(df.max_self == starting_x) & (df.max_other == starting_y) & (df.noise == 0),],
                       digit_sens).value_counts(['opt_x', 'opt_y'])
        counter = \
            round(df.loc[(df.max_self == starting_x) & (df.max_other == starting_y) & (df.noise == 0),],
                  2).value_counts(
                ['opt_x', 'opt_y']).iloc[0]
        modal_point = counts.idxmax()
        x0, y0 = modal_point
        slope = -starting_y / starting_x
        new_slope = slope_sens * slope
        x_int = x0 - y0 / new_slope
        y_int = y0 - new_slope * x0
        print(
            f"For budget set: ({starting_x}, {starting_y}) I found as modal point: {modal_point} with slope {slope}, count:{counter}")
        budgets.append((x_int, y_int))
        starting_x, starting_y = x_int, y_int
        starting_budget_set = (x_int, y_int)
        print(f"Starting x and y: {starting_x}, {starting_y}")
        print(f"Starting budget set: {starting_budget_set}\n")
        print("==============================================")
    return budgets
