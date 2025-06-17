import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from utils.test_rev_pref import compute_index, bin_index, count_violations
from fractions import Fraction
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display


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


def plot_noise(
        df: pd.DataFrame,
        utility_func,
        noise_sd: float,
        *,
        free_disposal: bool = True,
        save: bool = False,
        savepath: str = "",
        title: str = ""
):
    """
    Plot the optimal bundle and its noisy counterpart for a particular
    utility function / noise level combination.  The legend shows the
    parameter *distributions* that generated the draws.

    Parameters
    ----------
    df : DataFrame
        Must contain columns:
          - utility_func  (function object stored by your generator)
          - noise         (float sd used when adding noise)
          - opt_x, opt_y  optimal coordinates
          - noisy_x, noisy_y  noisy coordinates
          - max_self, max_other
    utility_func : callable
        The utility function whose points you wish to plot.
    param_distributions : dict[str, str]
        Keys: parameter names; Values: short human‑readable distribution
        descriptions to appear in the legend (e.g. "Beta(2, 5)").
    noise_sd : float
        The σ you want to inspect.
    free_disposal : bool, default True
        Only used to distinguish the file‑name if you decide to save.
    save : bool, default False
        Save the figure instead of showing it interactively.
    savepath : str, default ""
        Directory for saving (ignored unless `save=True`).
    title : str, default ""
    """
    # ---------------- filter the data we actually want ------------------
    subset = df[
        (df["utility_func"] == utility_func) &
        (df["noise"] == noise_sd)
        ]

    if subset.empty:
        raise ValueError("No rows match the requested utility function / noise SD.")

    # ---------------- legend label -------------------------------------
    opt_label = f"Optimal point"
    noi_label = f"Noisy point (σ={noise_sd})"

    # ---------------- plotting -----------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(
        subset["opt_x"], subset["opt_y"],
        edgecolors="black", label=opt_label
    )
    plt.scatter(
        subset["noisy_x"], subset["noisy_y"],
        edgecolors="black", color="red", label=noi_label
    )

    # budget lines
    for _, r in subset.iterrows():
        xs = np.linspace(0, r["max_self"], 100)
        ys = -(r["max_other"] / r["max_self"]) * xs + r["max_other"]
        plt.plot(xs, ys, color="gray", alpha=0.5, linestyle="dashed")

    plt.xlim(0, 160)
    plt.ylim(0, 160)
    plt.xlabel("Money for Self")
    plt.ylabel("Money for Other")
    if title:
        plt.title(title)
    else:
        plt.title(f"Noise analysis — {utility_func.__name__}")
    plt.legend()

    # ---------------- output -------------------------------------------
    suffix = "free_disposal" if free_disposal else "no_disposal"
    if save:
        fname = f"{savepath}/noise_sd_{noise_sd}_{suffix}.png"
        plt.savefig(fname, bbox_inches="tight")
    else:
        plt.show()


def plot_budget_sequence(budgets, opt_points, ax=None, title=None,
                         save_path=None, note=None):
    """
    Parameters
    ----------
    budgets : list[(float, float)]
        Each tuple (ms, mo) gives the self- and other-intercepts.
    opt_points : list[(float, float)]
        Optimal bundles corresponding to *budgets*.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on.  If *None*, a new figure/axis is created.
    title : str, optional
        Plot title.
    save_path : str, optional
        If given, save the figure there.
    note : str, optional
        Extra note shown below the plot.
    """
    # ---------------- set-up ----------------
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # ---------------- main drawing loop -----
    for (ms, mo), (x, y) in zip(budgets, opt_points):
        ax.plot([0, ms], [mo, 0], lw=1)
        ax.scatter([x], [y])

    # ---------------- axis limits & aspect --
    max_val = max(
        max(ms for ms, _ in budgets),
        max(mo for _, mo in budgets)
    )
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("self")
    ax.set_ylabel("other")

    # ---------------- titles / notes --------
    if title:
        ax.set_title(title)
    if note:
        fig.text(0.5, 0.01, note,
                 ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ---------------- save ------------------
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_index(
        raw_data: pd.DataFrame,
        *,
        index_type: str = "CCEI",
        title: str | None = None,
        save_path: str | None = None,
        note: str | None = None,
        group_by: str = "noise",
        bin_width: float = 0.05
):
    """
    Compute an index, bin it (via bin_index), and show a bar-plot.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Must include all columns required by `compute_index` *plus*
        the column named in *group_by* (e.g. 'noise' or 'pucktreatment').
    index_type : {"CCEI", "HMI"}, default "CCEI"
    title, save_path, note : str | None
        Standard plot customisations (unchanged from the original function).
    group_by : str, default "noise"
        Column used for the hue.  Typical values: "noise", "pucktreatment".
    bin_width : float, default 0.05
        Passed straight through to `bin_index`.
    """

    index_type = index_type.upper()

    # 1 - Compute the index. CCEI and HMI supported for now
    index_df = compute_index(raw_data, index_type=index_type)

    #  Attach the grouping variable if compute_index dropped it
    if group_by not in index_df.columns:
        if group_by in raw_data.columns:
            gmap = raw_data[["id", group_by]].drop_duplicates()
            index_df = index_df.merge(gmap, on="id", how="left")
        else:
            raise KeyError(f"'{group_by}' not in raw_data or index_df.")

    # 2 - bin + normalise
    binned = bin_index(
        index_df,
        index_type=index_type,
        group_by=group_by,
        bin_width=bin_width
    )

    # 3 - Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=binned,
        x="bin",
        y="fraction",
        hue=group_by,
        dodge=True,
        edgecolor="black"
    )

    plt.title(title or f"{index_type} distribution")
    plt.xlabel(f"{index_type} score (binned)")
    plt.ylabel("Fraction")
    plt.xticks(rotation=45, ha="right")

    # legend title
    if group_by == "noise":
        plt.legend(title="Noise σ")
    elif group_by == "pucktreatment":
        plt.legend(title="Treatment")
    else:
        plt.legend(title=group_by)

    if note:
        plt.figtext(0.5, -0.05, note,
                    wrap=True, ha="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_garp_trigger(data):
    """Plot how frequently each budget constraint triggers GARP violations.
    Args:
        data: a pandas dataframe obtained using simulate_dataset
    """
    results_df = count_violations(data)
    results_df["budget"] = results_df.apply(
        lambda row: f"({row['max_self']},{row['max_other']})", axis=1)

    plt.figure(figsize=(16, 6))
    sns.barplot(data=results_df.sort_values("garp_trigger_count", ascending=False),
                x="budget", y="garp_trigger_count", hue="noise")

    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Budget constraint (max_self, max_other)")
    plt.ylabel("Number of GARP violations triggered")
    plt.title("GARP‑violating budgets by noise level")
    plt.tight_layout()
    plt.show()


def plot_budget_constraints(
        downward_budgets,
        upward_pairs,
        *,
        x_max=None,
        show_projections=False
):
    """
    Draw downward‑ and upward‑sloping budget constraints.

    Parameters
    ----------
    downward_budgets : list[(int, int)]
        Each (max_self, max_other) gives one downward line:
        y = max_other - (max_other/max_self) * x
    upward_pairs : list[(Fraction|int|float, (int, int))]
        Output of find_integer_upward_slopes: every element is
        (slope, (x_int, y_int)), i.e. the upward line y = slope * x intersects
        some downward line at the lattice point (x_int, y_int).
    x_max : int | None
        Right‑hand limit of the plot.  Defaults to the largest x‑intercept.
    show_projections : bool
        If True, draw dotted projections of each intersection onto the axes.
    """
    # ---------- determine plotting window -----------------------------------
    if x_max is None:
        x_max = max(ms for ms, _ in downward_budgets)
    y_max = max(mo for _, mo in downward_budgets)
    limit = max(x_max, y_max)  # make axes equal scale

    fig, ax = plt.subplots(figsize=(8, 8))

    # ---------- plot downward lines -----------------------------------------
    for ms, mo in downward_budgets:
        ax.plot([0, ms], [mo, 0], linestyle="-",
                label=f"down ({ms},{mo})")

    # ---------- plot upward lines (segments to intersection) ----------------
    for slope, (xi, yi) in upward_pairs:
        m = float(slope)
        # segment from origin to intersection point
        ax.plot([0, xi], [0, yi], linestyle="--",
                label=f"up m={slope}")
        if show_projections:
            # vertical and horizontal projections (dotted)
            ax.plot([xi, xi], [0, yi], linestyle=":", linewidth=1)
            ax.plot([0, xi], [yi, yi], linestyle=":", linewidth=1)

    # ---------- decorate -----------------------------------------------------
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Self consumption (x)")
    ax.set_ylabel("Other consumption (y)")
    ax.set_title("Budget constraints with equal axes")
    ax.grid(True, ls=":")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_simple_budget_constraints(
        budgets: list[tuple[float, float]],
        *,
        title: str = None,
        colors: "str | list[str]" = 'gray',
        save: bool = False,
        savepath: str = "",
):
    """
    Plot budget constraints given a list of (max_self, max_other) tuples on a fixed scale.

    Parameters:
        budgets: List of tuples defining budget lines as (max_self, max_other).
        title: Optional title for the plot.
        colors: Single color or list of colors for each budget line. If a single color is
                provided, all lines use that color. If a list, its length must match budgets.
        save: If True, save the plot to `savepath` instead of showing.
        savepath: Full path (including filename) to save the figure.

    Both axes are fixed to range from 1 to 160.
    """

    if not budgets:
        raise ValueError("`budgets` list must contain at least one (max_self, max_other) tuple.")

    # Prepare color list
    if isinstance(colors, str):
        color_list = [colors] * len(budgets)
    else:
        if len(colors) != len(budgets):
            raise ValueError("Length of colors list must match number of budgets.")
        color_list = colors

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot each budget line with specified colors
    for (ms, mo), col in zip(budgets, color_list):
        ax.plot([0, ms], [mo, 0], linestyle='-', color=col, linewidth=1)

    # Fix the scale for both axes
    ax.set_xlim(1, 160)
    ax.set_ylim(1, 160)
    ax.set_xlabel('Self')
    ax.set_ylabel('Other')
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if save:
        # Ensure directory exists
        directory = os.path.dirname(savepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()


def plot_simple_upward_constraints(
        points: list[tuple[float, float]],
        *,
        title: str = None,
        colors: "str | list[str]" = 'gray',
        save: bool = False,
        savepath: str = "",
):
    """
    Plot upward-sloping constraints given a list of (max_self, max_other) tuples on a fixed scale.

    Each constraint is a straight line from the origin to the point (max_self, max_other).

    Parameters:
        points: List of tuples defining end-point of each upward line as (max_self, max_other).
        title: Optional title for the plot.
        colors: Single color or list of colors for each line. If a single color is
                provided, all lines use that color. If a list, its length must match points.
        save: If True, save the plot to `savepath` instead of showing.
        savepath: Full path (including filename) to save the figure.
    """
    if not points:
        raise ValueError("`points` list must contain at least one (max_self, max_other) tuple.")

    # Prepare color list
    if isinstance(colors, str):
        color_list = [colors] * len(points)
    else:
        if len(colors) != len(points):
            raise ValueError("Length of colors list must match number of points.")
        color_list = colors

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot each upward constraint
    for (ms, mo), col in zip(points, color_list):
        xs = [0, ms]
        ys = [0, mo]
        ax.plot(xs, ys, linestyle='-', color=col, linewidth=1)

    # Fix the scale for both axes
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 160)
    ax.set_xlabel('Self')
    ax.set_ylabel('Other')
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if save:
        # Ensure directory exists
        directory = os.path.dirname(savepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()


def _coerce_direction(val) -> int:
    """Return **‑1** or **1** depending on the sign of *val*.

    Accepts ints, floats, or strings such as "-1", "+1", "1.0".
    Raises *ValueError* for anything that does not evaluate to ±1.
    """
    try:
        num = float(val)
    except Exception as err:
        raise ValueError("direction value must be numeric ±1") from err
    if num < 0:
        return -1
    if num > 0:
        return 1
    raise ValueError("direction value cannot be zero — expected ±1")


def _draw_budget_line_matplotlib(
    ax: plt.Axes,
    max_self: float,
    max_other: float,
    direction: int | float | str,
    **kwargs,
):
    """Draw a single budget line (Matplotlib)."""
    direction = _coerce_direction(direction)
    if direction == -1:  # downward
        ax.plot([0, max_self], [max_other, 0], **kwargs)
    else:  # upward
        ax.plot([0, max_self], [0, max_other], **kwargs)


def _draw_budget_line_plotly(
    fig: go.Figure,
    max_self: float,
    max_other: float,
    direction: int | float | str,
    **kwargs,
):
    """Same as above for Plotly."""
    direction = _coerce_direction(direction)
    if direction == -1:
        x_coords, y_coords = [0, max_self], [max_other, 0]
    else:
        x_coords, y_coords = [0, max_self], [0, max_other]

    fig.add_trace(
        go.Scatter(x=x_coords, y=y_coords, mode="lines", hoverinfo="skip", **kwargs)
    )


# -----------------------------------------------------------------------------
# 1. Static: one treatment × one direction
# -----------------------------------------------------------------------------

def plot_static_choice_panel(
    df: pd.DataFrame,
    *,
    direction: int | float | str,
    treatment: str,
    figsize: tuple[int, int] = (6, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot **all** observed points & budgets for a given *direction* × *treatment*."""
    required_cols = {
        "noisy_x",
        "noisy_y",
        "direction",
        "max_self",
        "max_other",
        "pucktreatment",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing columns: {', '.join(missing)}")

    dir_int = _coerce_direction(direction)

    # ------------------------------------------------------------------
    subset = df[(df["pucktreatment"] == treatment) & (df["direction"].apply(_coerce_direction) == dir_int)]
    if subset.empty:
        raise ValueError("No data match the given direction/treatment filter.")

    fig, ax = plt.subplots(figsize=figsize)

    # Draw budgets & points -------------------------------------------
    for _, row in subset.iterrows():
        _draw_budget_line_matplotlib(
            ax,
            row["max_self"],
            row["max_other"],
            row["direction"],
            color="lightgrey",
            linewidth=1,
            zorder=1,
        )
        ax.scatter(row["noisy_x"], row["noisy_y"], s=20, zorder=2)

    # Cosmetics --------------------------------------------------------
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Own payoff (cents)")
    ax.set_ylabel("Other payoff (cents)")

    dir_name = "Downward" if dir_int == -1 else "Upward"
    ax.set_title(f"{treatment.capitalize()} group — {dir_name} constraints")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


# ----------------------------------------------------------------------------
# 2. Interactive dashboard (Jupyter/VSCode)
# ----------------------------------------------------------------------------

def plot_subject_choices_dashboard(df: pd.DataFrame) -> None:
    """Launch an interactive dashboard for per‑subject choices.

    Widgets:
      • **Subject ID** — dropdown of unique *id* values
      • **Direction** — dropdown (‑1 / 1)
      • **Choice #** — dropdown (None ⇒ all choices)

    Now robust to *direction* stored as floats or strings and always renders
    upward constraints correctly.
    """
    # Guard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    required_cols = {
        "id",
        "direction",
        "max_self",
        "max_other",
        "noisy_x",
        "noisy_y",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing columns: {', '.join(missing)}")

    # Ensure we have numeric direction for look‑ups --------------------
    df = df.copy()
    df["_dir_int"] = df["direction"].apply(_coerce_direction)

    # Widget set‑up -----------------------------------------------------
    id_selector = widgets.Dropdown(options=sorted(df["id"].unique()), description="Subject")
    direction_selector = widgets.Dropdown(options=[-1, 1], description="Direction")
    choice_selector = widgets.Dropdown(description="Choice #")
    output = widgets.Output()

    def _filter_df() -> pd.DataFrame:
        return df[(df["id"] == id_selector.value) & (df["_dir_int"] == int(direction_selector.value))]

    def _update_choice_options(*_):
        sub = _filter_df()
        n = len(sub)
        if n:
            choice_selector.options = [None] + list(range(n))
            if choice_selector.value not in choice_selector.options:
                choice_selector.value = None
        else:
            choice_selector.options = (None,)
            choice_selector.value = None

    def _make_plot(sub: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        # Budgets ------------------------------------------------------
        for _, row in sub.iterrows():
            _draw_budget_line_plotly(
                fig,
                row["max_self"],
                row["max_other"],
                row["_dir_int"],
                line=dict(color="lightgrey"),
                showlegend=False,
            )
        # Points -------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=sub["noisy_x"],
                y=sub["noisy_y"],
                mode="markers",
                marker=dict(size=9),
                name="Choice",
            )
        )
        dir_name = "Downward" if int(direction_selector.value) == -1 else "Upward"
        fig.update_xaxes(range=[0, 2000], title="Own payoff (cents)")
        fig.update_yaxes(range=[0, 2000], title="Other payoff (cents)")
        fig.update_layout(height=650, width=650, title=f"Subject {id_selector.value} — {dir_name} budgets")
        return fig

    def _redraw(*_):
        output.clear_output(wait=True)
        sub = _filter_df().reset_index(drop=True)
        if choice_selector.value is not None:
            if len(sub) == 0:
                sub = pd.DataFrame()
            elif 0 <= choice_selector.value < len(sub):
                sub = sub.iloc[[choice_selector.value]]
            else:
                sub = pd.DataFrame()
        with output:
            if sub.empty:
                print("No data for this combination.")
            else:
                _make_plot(sub).show()

    # Connect callbacks -----------------------------------------------
    id_selector.observe(_update_choice_options, names="value")
    direction_selector.observe(_update_choice_options, names="value")

    id_selector.observe(_redraw, names="value")
    direction_selector.observe(_redraw, names="value")
    choice_selector.observe(_redraw, names="value")

    # Initial render ---------------------------------------------------
    _update_choice_options()
    _redraw()

    display(widgets.VBox([id_selector, direction_selector, choice_selector, output]))