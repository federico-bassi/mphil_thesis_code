"""
Helper functions to retrieve data from mysql, clean the table, and produce tables.
"""
import mysql.connector
import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np
from pathlib import Path
from typing import Mapping


def fetch_table(database, table):
    """ Retrieve all the rows from the table and database specified, returns a pandas Dataframe."""
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database=database
    )
    cursor = mydb.cursor()
    cursor.execute(f"SELECT * FROM {table}")
    rows = cursor.fetchall()
    columns = [i[0] for i in cursor.description]
    cursor.close()
    mydb.close()
    return pd.DataFrame(rows, columns=columns)


def tidy_questionnaire(df: pd.DataFrame) -> pd.DataFrame:
    """Convert questionnaire rows into analysable columns, translated to english and in the correct format."""
    out = df.copy()

    # --- recodes variables --------------------------------------------------e
    out["age"] = pd.to_numeric(out["leeftijd"], errors="coerce")  # age
    out["female"] = (out["sexe"].str.lower() == "vrouw").astype(int)  # sex
    out["econ_student"] = (out["studie"] == "econ").astype(int)  # field of study
    out["creed_exp"] = (out["recivide"].str.lower() != "no").astype(int)  # experience with CREED
    out["international"] = (out["nationaliteit"] == "Inter").astype(int)  # international student

    # --- year fields --------------------------------------------------------
    def _extract_year(x, default=np.nan):
        if pd.isna(x):
            return np.nan
        import re

        m = re.search(r"\b(19|20)\d{2}\b", str(x))
        return int(m.group()) if m else default

    out["year_begin"] = out["jaarbegin"].apply(_extract_year)
    out["year_end"] = out["jaarafstuderen"].apply(_extract_year)

    # --- Likert ratings -----------------------------------------------------
    likert = [
        "mdifficult", "mfun", "mstressfull",
        "gdifficult", "gfun", "gstressfull",
    ]
    out[likert] = out[likert].apply(pd.to_numeric, errors="coerce")

    return out


def describe_by_group_to_latex(
    df: pd.DataFrame,
    vars_to_check: Mapping[str, str],
    *,
    group_col: str = "group",
    filename: str = "describe_by_group_table.tex",
    caption: str = "Descriptive statistics by group",
    label: str = "tab:desc_stats",
    note: str | None = None,
    float_format: str = ".2f",
) -> str:
    """
    Build a LaTeX table with means (SD) for each variable by treatment status
    and a two-sided Mann–Whitney U-test p-value.
    """

    # --- Sanity check -----------------------------------------------------
    n_treat = (df[group_col] == "treatment").sum()
    n_ctrl = (df[group_col] == "control").sum()
    if n_treat == 0 or n_ctrl == 0:
        raise ValueError(
            f"'{group_col}' must contain at least one 'treatment' and one "
            f"'control' row (found {n_treat} / {n_ctrl}). Check coding."
        )

    # --- Build rows -------------------------------------------------------
    rows: list[tuple[str, str, str, str]] = []
    for nice, col in vars_to_check.items():
        t = df.loc[df[group_col] == "treatment", col].dropna()
        c = df.loc[df[group_col] == "control", col].dropna()

        mean_t, sd_t = t.mean(), t.std(ddof=1)          # sample SD
        mean_c, sd_c = c.mean(), c.std(ddof=1)

        p_val = (
            mannwhitneyu(t, c, alternative="two-sided").pvalue
            if len(t) and len(c)
            else np.nan
        )

        ctrl_cell  = f"{mean_c:{float_format}} ({sd_c:{float_format}})"
        treat_cell = f"{mean_t:{float_format}} ({sd_t:{float_format}})"
        p_cell     = f"{p_val:{float_format}}" if pd.notna(p_val) else ""

        rows.append((nice, ctrl_cell, treat_cell, p_cell))

    rows.append(("Observations", f"{n_ctrl}", f"{n_treat}", ""))

    if note is None:
        note = (
            "Note: Cells show means with standard deviations in parentheses. "
            "The last column reports two-sided Mann–Whitney $U$-test "
            "$p$-values. The 'Observations' row indicates non-missing cases "
            "per group."
        )

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        fr"  \caption{{{caption}}}",
        fr"  \label{{{label}}}",
        r"  \sisetup{table-format=2.2,table-number-alignment=center}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{lccc}",
        r"      \toprule",
        r"      Variable & Control & Treatment & Mann--Whitney $p$ \\",
        r"      \midrule",
    ]
    for variable, ctrl_cell, treat_cell, p_cell in rows:
        lines.append(fr"      {variable} & {ctrl_cell} & {treat_cell} & {p_cell} \\")
    lines += [
        r"      \bottomrule",
        r"    \end{tabular}",
        r"    \begin{tablenotes}[flushleft]",
        fr"      \footnotesize {note}",
        r"    \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]

    latex_str = "\n".join(lines)
    Path(filename).write_text(latex_str, encoding="utf-8")
    return latex_str


def concat_with_ids(
        df_session1: pd.DataFrame,
        df_session2: pd.DataFrame,
        *,
        ppn_col: str = "ppnr",
        id_col: str = "id",
        keep_ppnr: bool = True
) -> pd.DataFrame:
    """
    Concatenate two session data-frames with a **continuous, gap-free** `id` column.

    Steps
    -----
    1.  Collect the unique `ppnr`s of session-1, sort them, and
        map them to 1…N  (fills any gaps in the original numbering).
    2.  Collect the unique `ppnr`s of session-2, sort them, and
        map them to N+1…M   (no overlap with session-1).
    3.  Apply the two mappings, add the `id` column, and `pd.concat`.
    4.  Assert that the resulting `id` column is unique.

    Parameters
    ----------
    df_session1, df_session2 : pd.DataFrame
        Data from session 1 and session 2, each containing a `ppnr` column.
    ppn_col : str, default "ppnr"
        Name of the participant-number column in the inputs.
    id_col : str, default "id"
        Name of the new identifier column to create.
    keep_ppnr : bool, default True
        If *False*, the original `ppnr` column is dropped from the output.

    Returns
    -------
    pd.DataFrame
        A single dataframe with a continuous, `id` column.
    """

    uniq1 = pd.Series(df_session1[ppn_col].unique()).sort_values()
    map1 = {pp: i + 1 for i, pp in enumerate(uniq1)}

    start = len(map1) + 1
    uniq2 = pd.Series(df_session2[ppn_col].unique()).sort_values()
    map2 = {pp: start + i for i, pp in enumerate(uniq2)}

    s1 = df_session1.copy()
    s2 = df_session2.copy()

    s1[id_col] = s1[ppn_col].map(map1)
    s2[id_col] = s2[ppn_col].map(map2)

    if not keep_ppnr:
        s1.drop(columns=[ppn_col], inplace=True)
        s2.drop(columns=[ppn_col], inplace=True)

    out = pd.concat([s1, s2], ignore_index=True)

    #assert out[id_col].is_unique, "Duplicate IDs created — check ppnr data!"

    cols = [id_col] + [c for c in out.columns if c != id_col]
    out = out.loc[:, cols]

    return out


__all__ = ["describe_by_group_to_latex", "fetch_table", "tidy_questionnaire", "concat_with_ids"]
