import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import revpref as rp
import os


def generate_revealed_preference_arrays(data):
    # Create an array of prices where the price of x is 1 and the price of y is max_self/max_other ("price of giving")
    p = np.column_stack((np.ones(len(data)), data["max_self"] / data["max_other"]))

    # Create an array of quantities
    q = np.column_stack((data['noisy_x'], data['noisy_y']))
    return p, q


def compute_ccei_distribution(data):
    results = []
    for (id_val, noise_level), group in data.groupby(['id', 'noise']):
        p, q = generate_revealed_preference_arrays(group)
        pref = rp.RevealedPreference(p, q)
        try:
            ccei = pref.ccei(method='bisection', tol=1e-9)
        except Exception:
            ccei = np.nan
        results.append({'id': id_val, 'noise': noise_level, 'CCEI': ccei})
    return pd.DataFrame(results)


def bin_ccei_data(ccei_data, bin_width=0.05):
    breaks = np.arange(0, 1 + bin_width, bin_width)
    labels = [f"[{round(breaks[i], 2)}, {round(breaks[i + 1], 2)})" for i in range(len(breaks) - 1)]
    ccei_data['bin'] = pd.cut(ccei_data['CCEI'], bins=breaks, labels=labels, include_lowest=True, right=False)
    ccei_data['noise'] = pd.Categorical(ccei_data['noise'],
                                        categories=sorted(ccei_data['noise'].unique(), reverse=True))
    binned = ccei_data.groupby(['bin', 'noise']).size().reset_index(name='count')
    return binned


def plot_binned_ccei(binned_data, title, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=binned_data, x='bin', y='count', hue='noise', dodge=True, edgecolor='black')
    plt.title(title)
    plt.xlabel("CCEI Score (Binned)")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Noise Std")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


# Example usage:
if __name__ == "__main__":
    path = "/Users/federicobassi/Desktop/TI.nosync/MPhil_Thesis/"

    data_free_disposal = pd.read_excel(path + "/simulated_data/results_free_disposal.xlsx")
    ccei_free_disposal = compute_ccei_distribution(data_free_disposal)
    binned_free_disposal = bin_ccei_data(ccei_free_disposal)
    plot_binned_ccei(binned_free_disposal, "CCEI distribution, free disposal", save_path= path+"/python_no_disposal.png")
