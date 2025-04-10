from scipy.optimize import minimize
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, LinearConstraint

np.random.seed(123)


def fehr_schmidt_utility(x, y, alpha=0.5, beta=0.5):
    """Computes Fehr-Schmidt utility, incorporating inequality aversion parameters."""
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


def budget_constraint(max_self, max_other, x):
    """Defines the budget constraint given maximum resources."""
    return max_other - (max_other / max_self) * x


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


def data_generation_random_parameters(budgets, utility_func, utility_label, param_distributions, n_samples=50):
    """Generates a DataFrame with budget constraints and utility functions with random parameters.

    Parameters:
    budgets (list of tuples): List of (max_self, max_other) budget constraints.
    utility_func (function): Utility function with variable parameters.
    utility_label (str): Label identifying the utility function.
    param_distributions (dict): Dictionary with parameter names as keys and distributions as values (callable).
    n_samples (int): Number of samples to draw from the parameter distributions.

    Returns:
    pd.DataFrame: DataFrame containing budget constraints and corresponding utility functions.
    """
    data = []
    for max_self, max_other in budgets:
        for i in range(n_samples):
            id = i + 1

            # Sample parameters
            sampled_params = {key: dist() for key, dist in param_distributions.items()}

            # Define a concrete function with fixed parameters
            def utility_fixed(x, y, sampled_params=sampled_params):
                return utility_func(x, y, **sampled_params)

            # Store the function itself (not a lambda)
            data.append((id, max_self, max_other, utility_fixed, utility_label))

    return pd.DataFrame(data, columns=['id', 'max_self', 'max_other', 'utility_func', 'utility_label'])


def maximize_utility(df):
    """Optimizes utility functions subject to budget constraints.

    Parameters:
    df (pd.DataFrame): DataFrame containing budget constraints and utility functions.

    Returns:
    pd.DataFrame: DataFrame containing optimization results.
    """
    results = []

    for _, row in df.iterrows():
        # Define the negative utility for minimization
        def neg_utility(z):
            return -row['utility_func'](z[0], z[1])

        # Budget bounds
        bounds = [(0, row['max_self']), (0, row['max_other'])]

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


def maximize_utility_global(df):
    """Optimizes utility functions using differential_evolution with linear budget constraints,
    then refines results with local SLSQP optimization."""

    results = []

    for _, row in df.iterrows():
        print(f"Agent {row['id']} is maximizing utility!")

        max_self, max_other = row['max_self'], row['max_other']

        # Objective: negative utility for minimization
        def neg_utility(z):
            return -row['utility_func'](z[0], z[1])

        # Bounds on x and y
        bounds = [(0, max_self), (0, max_other)]

        # Constraint: y <= budget line
        slope = max_other / max_self if max_self != 0 else 0
        linear_constraint = LinearConstraint([[slope, 1]], -np.inf, max_other)

        # Global optimization with DE
        result_de = differential_evolution(
            neg_utility,
            bounds,
            constraints=(linear_constraint,),
            maxiter=10000,
            tol=1e-10,
            popsize=25,
            polish=True,
            seed=123,
            strategy='best1bin',
            updating='deferred'
        )

        # Local refinement with SLSQP
        result_local = minimize(
            neg_utility,
            result_de.x,  # start from DE result
            method="SLSQP",
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda z: budget_constraint(max_self, max_other, z[0]) - z[1]},
            # options={'ftol': 1e-15, 'maxiter': 1000}
        )

        # Final choice: prefer local if successful, else fallback to DE
        if result_local.success:
            final_x, final_y = result_local.x
            success = True
        else:
            final_x, final_y = result_de.x
            success = result_de.success

        results.append((
            row["id"],
            max_self,
            max_other,
            float(final_x),
            float(final_y),
            row['utility_label'],
            success
        ))

    return pd.DataFrame(results, columns=["id", 'max_self', 'max_other', 'opt_x', 'opt_y', 'utility_label', "success"])


def add_noise(df, mean=0, std=5, free_disposal=True):
    """
    Adds noise to optimal values. If free_disposal is True, noise is added
    along the budget constraint only (tangent direction). Otherwise, noise
    can move both along and inward from the budget line.

    Parameters:
    df (pd.DataFrame): Must contain 'opt_x', 'opt_y', 'max_self', and 'max_other'.
    mean (float): Mean of the noise (not used).
    std (float): Standard deviation of the noise.
    free_disposal (bool): If True, noise moves only along budget line. Else, also inward.

    Returns:
    pd.DataFrame: With new columns 'noisy_x', 'noisy_y', and 'noise'.
    """
    df = df.copy()

    for index, row in df.iterrows():
        opt_point = np.array([row['opt_x'], row['opt_y']])
        max_self, max_other = row["max_self"], row["max_other"]

        # Define orthogonal and tangent directions based on price vector
        p1 = 1
        p2 = max_self / max_other
        orthogonal = np.array([p1, p2])
        orthogonal /= np.linalg.norm(orthogonal)

        tangent = np.array([-p2, p1])
        tangent /= np.linalg.norm(tangent)

        # Generate displacement
        t = np.random.normal()
        # only inward movement if not free_disposal
        o = 0 if free_disposal else -abs(np.random.normal())
        displacement = std * (t * tangent + o * orthogonal)

        noisy_point = opt_point + displacement
        df.loc[index, "noisy_x"] = max(float(noisy_point[0]), 0)
        df.loc[index, "noisy_y"] = max(float(noisy_point[1]), 0)

    df["noise"] = std
    return df


def compute_std(df, budget_constraints):
    """
    Computes the standard deviation of "opt_x" or "opt_y" for each budget constraint. Outputs a dictionary.

    :param df: the dataframe
    :param budget_constraints: a list of tuples representing budget constraints

    :return: a dictionary with tuples as keys (budget constraints) and std values as values.
    """

    # Initialize an empty dictionary to store results
    std_by_budget = {}

    # Loop through each (max_self, max_other) pair
    for max_self, max_other in budget_constraints:
        # Filter the DataFrame for that specific budget constraint
        subset = df[(df.max_self == max_self) & (df.max_other == max_other)]

        # Calculate the std of opt_x for that subset
        std_x = round(subset["opt_x"].std(), 2)

        # Store in the dictionary using the budget pair as the key
        std_by_budget[(max_self, max_other)] = std_x

    return std_by_budget


if __name__ == '__main__':
    # Parameters
    alpha_fs, beta_fs = 0.5, 0.5
    altruism_parameter = 0.5
    alpha_cd, beta_cd = 0.5, 0.5
    alpha_ces, rho_ces = 0.5, 0.1
    mean_noise, sd_noise = 0, 5
    param_distribution_ces = {'alpha_ces': lambda: np.random.uniform(0.5, 0.9),
                              'rho_ces': lambda: np.random.uniform(-10, 0.9)}
    samples = 200
    path = "/Users/federicobassi/Desktop/TI.nosync/MPhil_Thesis/simulated_data"

    # Define budget constraints
    andreoni_miller_budgets = [(120, 40), (40, 120), (120, 60), (60, 120), (150, 75),
                               (75, 150), (60, 60), (100, 100), (80, 80), (160, 40), (40, 160)]

    new_budgets = [(120, 40), (40, 120), (120, 60), (60, 120), (150, 75), (75, 150), (60, 60), (100, 100),
                   (80, 80), (160, 40), (40, 160),
                   (120, 120), (100, 150), (150, 100), (40, 40), (30, 60), (60, 30), (50,50), (30, 150), (150,30)]

    # Define utility functions with parameterized lambda functions
    utility_funcs = [
        (ces, {'alpha_ces': 0.5, 'rho_ces': -0.5}, "CES (α=0.5, ρ=-0.5)"),
        (ces, {'alpha_ces': 0.9, 'rho_ces': -0.5}, "CES (α=0.9, ρ=-0.5)"),
        (ces, {'alpha_ces': 0.5, 'rho_ces': 0.5}, "CES (α=0.5, ρ=0.5)"),
        (ces, {'alpha_ces': 0.9, 'rho_ces': 0.5}, "CES (α=0.9, ρ=0.5)"),
    ]

    # Generate data
    df_budgets_utility_ces = data_generation_random_parameters(budgets=new_budgets, utility_func=ces,
                                                               utility_label="CES",
                                                               param_distributions=param_distribution_ces,
                                                               n_samples=samples)

    # df_budgets_utility = data_generation(budgets=andreoni_miller_budgets, utility_funcs=utility_funcs)

    # Perform optimization
    opt_results = maximize_utility(df_budgets_utility_ces)
    opt_results.to_excel(path + "/optimization_extended_budgets.xlsx")

    # Compute the standard deviations
    # dict_std = compute_std(df=opt_results, budget_constraints=andreoni_miller_budgets)
    # print(dict_std)

    # Add noise
    opt_results = pd.read_excel(path + "/optimization_extended_budgets.xlsx")
    std_interval = [0, 1, 2, 3, 5, 10]
    noisy_data_no_disposal = pd.DataFrame()
    noisy_data_free_disposal = pd.DataFrame()
    for sd_noise in std_interval:
        data_no_disposal = add_noise(opt_results, mean=mean_noise, std=sd_noise, free_disposal=False)
        data_disposal = add_noise(opt_results, mean=mean_noise, std=sd_noise, free_disposal=True)
        noisy_data_no_disposal = pd.concat([noisy_data_no_disposal, data_no_disposal], ignore_index=True)
        noisy_data_free_disposal = pd.concat([noisy_data_free_disposal, data_disposal], ignore_index=True)

    # Save results to Excel
    noisy_data_no_disposal.to_excel(path + "/results_no_disposal.xlsx", index_label="id", index=False)
    noisy_data_free_disposal.to_excel(path + "/results_free_disposal.xlsx", index_label="id", index=False)
