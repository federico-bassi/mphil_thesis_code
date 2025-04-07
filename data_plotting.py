import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px


def plot_static_budget_constraints(df, save=False, savepath=""):
    """Plots static budget constraints with optimization points using Matplotlib."""
    plt.figure(figsize=(8, 6))

    # Get unique utility labels for coloring
    utility_labels = df['utility_label'].unique()
    colors = sns.color_palette("tab10", len(utility_labels))
    color_dict = dict(zip(utility_labels, colors))

    # Plot budget constraints
    for _, row in df.iterrows():
        x_vals = np.linspace(0, row['max_self'], 100)
        y_vals = -(row['max_other'] / row['max_self']) * x_vals + row['max_other']
        plt.plot(x_vals, y_vals, color='gray', alpha=0.5, linestyle='dashed')

    # Plot optimal points
    for label in utility_labels:
        subset = df[df['utility_label'] == label]
        plt.scatter(subset['opt_x'], subset['opt_y'], color=color_dict[label], label=label, edgecolors='black')

    plt.xlim(-1, 161)
    plt.ylim(-1, 161)
    plt.xlabel('Money for Self')
    plt.ylabel('Money for Other')
    plt.legend(title='Utility Function')
    plt.title('Optimization Results')
    if save:
        plt.savefig(savepath + "/opt_results.png")
    else:
        plt.show()


def plot_interactive_budget_constraints(df):
    """Plots interactive budget constraints with optimization points using Plotly."""
    fig = px.line()

    # Plot budget constraints
    for _, row in df.iterrows():
        x_vals = np.linspace(0, row['max_self'], 100)
        y_vals = -(row['max_other'] / row['max_self']) * x_vals + row['max_other']
        fig.add_scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='gray', dash='dash'), showlegend=False)

    # Plot optimal points
    fig_opt = px.scatter(df, x='opt_x', y='opt_y', color='utility_label', symbol='utility_label',
                         labels={'opt_x': 'Money for Self', 'opt_y': 'Money for Other',
                                 'utility_label': 'Utility Function'},
                         title='Optimization Results',
                         size_max=12)

    # Combine both plots
    for trace in fig_opt.data:
        trace.marker.size = 12
        fig.add_trace(trace)

    fig.update_xaxes(range=[-1, 161])
    fig.update_yaxes(range=[-1, 161])
    fig.show()


def plot_noise(df, utility_function, noise_sd, free_disposal=True, save=False, savepath=""):
    """Plots the optimization point and the noisy point for a given utility function."""
    subset = df[(df['utility_label'] == utility_function) & (df["noise"] == noise_sd)]

    # Plot optimal and noisy points
    plt.figure(figsize=(8, 6))
    plt.scatter(subset['opt_x'], subset['opt_y'], color='blue', label='Optimal Point', edgecolors='black')
    plt.scatter(subset['noisy_x'], subset['noisy_y'], color='red', label='Noisy Point', edgecolors='black')

    # Plot budget constraints
    for _, row in df.iterrows():
        x_vals = np.linspace(0, row['max_self'], 100)
        y_vals = -(row['max_other'] / row['max_self']) * x_vals + row['max_other']
        plt.plot(x_vals, y_vals, color='gray', alpha=0.5, linestyle='dashed')

    plt.xlim(0, 160)
    plt.ylim(0, 160)
    plt.xlabel('Money for Self')
    plt.ylabel('Money for Other')
    plt.legend()
    plt.title(f'Noise Analysis for {utility_function}, SD: {noise_sd}')

    save_str = ""
    if free_disposal:
        save_str = "free_disposal"
    else:
        save_str = "no_disposal"

    if save:
        plt.savefig(savepath + f"/noise_sd_{noise_sd}_{save_str}.png")
    else:
        plt.show()


if __name__ == '__main__':
    path = "/Users/federicobassi/Desktop/TI.nosync/MPhil_Thesis/simulated_data"
    img_path = "/Users/federicobassi/Desktop/TI.nosync/MPhil_Thesis/plots"
    data_no_disposal = pd.read_excel(path + "/results_no_disposal.xlsx", index_col=0)
    data_free_disposal = pd.read_excel(path + "/results_free_disposal.xlsx", index_col=0)

    # EXECUTION
    plot_static_budget_constraints(data_no_disposal, save=True, savepath=img_path)

    for noise in [12, 24, 48]:
        plot_noise(data_no_disposal, "CES", noise_sd=noise, free_disposal=False, save=False, savepath=img_path)
        plot_noise(data_free_disposal, "CES", noise_sd=noise, free_disposal=True, save=False, savepath=img_path)

    # plot_noise(data, "CES", noise_sd=10, save=False, savepath=img_path)
