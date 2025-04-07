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


def data_generation(budgets, utility_functions):
    """Generates a DataFrame with budget constraints and utility functions.

    Parameters:
    budgets (list of tuples): List of (max_self, max_other) budget constraints.
    utility_functions (list of tuples): List of (function, label) utility functions.

    Returns:
    pd.DataFrame: DataFrame containing budget constraints and corresponding utility functions.
    """
    data = []
    ind = 1
    for max_self, max_other in budgets:
        for func, label in utility_functions:
            data.append((ind, max_self, max_other, func, label))
            ind += 1
    return pd.DataFrame(data, columns=["id", "max_self", 'max_other', 'utility_func', 'utility_label'])


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
        initial_guess = [row['max_self'] / 10, row['max_other'] / 10]

        # Run constrained optimization
        result = minimize(
            neg_utility,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 10000}
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

    return pd.DataFrame(results, columns=["id", 'max_self', 'max_other', 'opt_x', 'opt_y', 'utility_label'])


def maximize_utility_global(df):
    """Optimizes utility functions using differential_evolution with linear budget constraints.

    Parameters:
    df (pd.DataFrame): DataFrame with budget constraints and utility functions.

    Returns:
    pd.DataFrame: DataFrame containing optimization results.
    """
    results = []

    for _, row in df.iterrows():
        print("Agent "+ str(row["id"]) +" is maximizing utility!")
        max_self, max_other = row['max_self'], row['max_other']

        # Objective function to minimize (negative utility)
        def neg_utility(z):
            return -row['utility_func'](z[0], z[1])

        # Bounds on x and y
        bounds = [(0, max_self), (0, max_other)]

        # Budget constraint: y <= -(max_other/max_self)*x + max_other
        slope = max_other / max_self if max_self != 0 else 0
        constraint = LinearConstraint([[slope, 1]], -np.inf, max_other)

        # Run global optimization
        result = differential_evolution(
            neg_utility,
            bounds,
            constraints=(constraint,),
            maxiter=10000,
            tol=1e-9,
            polish=True,
            seed=123
        )

        results.append((
            row["id"],
            max_self,
            max_other,
            float(result.x[0]),
            float(result.x[1]),
            row['utility_label'],
            result.success
        ))

    return pd.DataFrame(results, columns=["id", 'max_self', 'max_other', 'opt_x', 'opt_y', 'utility_label', "success"])


def add_noise(df, mean=0, std=5, free_disposal=True):
    """Adds negative noise to optimal values while ensuring non-negativity.

    Parameters:
    df (pd.DataFrame): DataFrame containing optimized values.
    mean (float): Mean of the normal distribution for noise.
    std (float): Standard deviation of the normal distribution for noise.
    free_disposal(boolean): Specifies how the noise should be added. If true, agent chooses a point in the interior,
                            if false, agent is constrained on the budget set.

    Returns:
    pd.DataFrame: DataFrame with added 'noisy_x' and 'noisy_y' columns.
    """
    df = df.copy()
    if free_disposal:
        noise_x = -np.abs(np.random.normal(mean, std, size=len(df)))
        noise_y = -np.abs(np.random.normal(mean, std, size=len(df)))
        df['noisy_x'] = np.where(df['opt_x'] > 0, df['opt_x'] + noise_x, df['opt_x'])
        df['noisy_y'] = np.where(df['opt_y'] > 0, df['opt_y'] + noise_y, df['opt_y'])
        df['noisy_x'] = df['noisy_x'].clip(lower=0)
        df['noisy_y'] = df['noisy_y'].clip(lower=0)
    else:
        for index, row in df.iterrows():
            m = row["max_other"]/row["max_self"]
            dx = np.random.normal(0, std/np.sqrt(1 + m**2))
            noisy_x = np.clip(row['opt_x'] + dx, 0, row["max_self"])
            df.loc[index, "noisy_x"] = noisy_x
            df.loc[index, "noisy_y"] = budget_constraint(row["max_self"], row["max_other"], noisy_x)

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
    param_distribution_ces = {'alpha_ces': lambda: np.random.uniform(0.5, 0.7), 'rho_ces': lambda: np.random.uniform(0.2, 0.7)}
    samples = 100
    path = "/Users/federicobassi/Desktop/TI.nosync/MPhil_Thesis/simulated_data"

    # Define budget constraints
    andreoni_miller_budgets = [(120, 40), (40, 120), (120, 60), (60, 120), (150, 75), (75, 150), (60, 60), (100, 100), (80, 80), (160, 40), (40, 160)]

    new_budgets = [(120, 40), (40, 120), (120, 60), (60, 120), (150, 75), (75, 150), (60, 60), (100, 100),
                   (80, 80), (160, 40), (40, 160), (50, 10), (10, 50), (12, 10), (10, 12), (8, 16), (16,8),
                   (100, 80), (80,100), (40, 200), (200, 40), (20, 160), (160, 20), (35, 35)]

    # Define utility functions with parameterized lambda functions
    utility_functions = [
        (lambda x, y: fehr_schmidt_utility(x, y, alpha=alpha_fs, beta=beta_fs), "FS"),
        (lambda x, y: altruistic_utility(x, y, theta=altruism_parameter), "A"),
        (lambda x, y: selfish_utility(x, y), "S"),
        (lambda x, y: cobb_douglas(x, y, alpha=alpha_cd, beta=beta_cd), "CD"),
        (lambda x, y: ces(x, y, alpha_ces=alpha_ces, rho_ces=rho_ces), "CES")
    ]

    # Generate data
    df_budgets_utility_ces = data_generation_random_parameters(budgets=new_budgets, utility_func=ces,
                                                               utility_label="CES",
                                                               param_distributions=param_distribution_ces,
                                                               n_samples=samples)

    # Perform optimization
    opt_results = maximize_utility_global(df_budgets_utility_ces)
    opt_results.to_excel(path+"/optimization.xlsx")

    # Compute the standard deviations
    #dict_std = compute_std(df=opt_results, budget_constraints=andreoni_miller_budgets)
    #print(dict_std)

    # Add noise
    opt_results = pd.read_excel(path+"/optimization.xlsx")
    std_interval = [0, 2, 4, 6]
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
