"""
Shared plotting utilities for ASR leaderboard Pareto frontier plots.
"""

import pandas as pd
import matplotlib.pyplot as plt


def compute_pareto(df, x_col, y_col, x_goal="min", y_goal="min", y_fact=1.0):
    """
    Compute Pareto frontier.
    x_goal, y_goal ∈ {"min", "max"}
    """
    assert x_goal in {"min", "max"}
    assert y_goal in {"min", "max"}

    # Sort so we sweep correctly
    ascending = (x_goal == "min")
    df_sorted = df.sort_values(x_col, ascending=ascending)

    pareto = []

    if y_goal == "min":
        best_y = float("inf")
        for _, row in df_sorted.iterrows():
            if row[y_col] * y_fact < best_y:
                pareto.append(row)
                best_y = row[y_col] * y_fact
    else:  # y_goal == "max"
        best_y = -float("inf")
        for _, row in df_sorted.iterrows():
            if row[y_col] * y_fact > best_y:
                pareto.append(row)
                best_y = row[y_col] * y_fact

    return pd.DataFrame(pareto)


def plot_wer_tradeoff(
    df,
    x_col,
    y_col,
    label_col,
    x_label,
    y_label,
    output_file,
    x_goal="min",
    y_goal="min",
    y_log_scale=True,
    x_lim=None,
    y_lim=None,
    y_fact=1.0,
    font_size=18,
    highlight_model=None,
    title=None,
):
    # -----------------------
    # Drop rows with missing data for this plot
    # -----------------------
    df_plot = df[[x_col, y_col, label_col]].dropna()
    
    # Drop rows where y_col is -1 (missing/unknown values)
    df_plot = df_plot[df_plot[y_col] != -1]

    if df_plot.empty:
        print(f"Warning: no valid data for plot '{output_file}'")
        return

    pareto_df = compute_pareto(df_plot, x_col, y_col, x_goal=x_goal, y_goal=y_goal, y_fact=y_fact)
    pareto_labels = set(pareto_df[label_col].values)

    # Filter points within limits before plotting
    if x_lim is not None:
        df_plot = df_plot[(df_plot[x_col] >= x_lim[0]) & (df_plot[x_col] <= x_lim[1])]
    if y_lim is not None:
        df_plot = df_plot[(df_plot[y_col] * y_fact >= y_lim[0]) & (df_plot[y_col] * y_fact <= y_lim[1])]

    # Use a nice style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 8))

    # Set y_offset before using it in filtering
    y_offset = 1.35 if y_label.strip().lower() == 'rtfx' else 1.0

    # Margin for label/marker cutoff (as fraction of axis range)
    margin_x = 0.07 if x_lim is not None and y_label.strip().lower() == 'rtfx' else (0.02 if x_lim is not None else 0.0)
    margin_y = 0.02 if y_lim is not None else 0.0
    x_max = x_lim[1] if x_lim is not None else df_plot[x_col].max()
    y_max = y_lim[1] if y_lim is not None else (df_plot[y_col] * y_fact).max()

    def marker_within_bounds(x, y):
        return (x < x_max * (1 - margin_x)) and (y < y_max * (1 - margin_y))

    # Separate Pareto and non-Pareto points for different styling, filter for RTFx margin
    pareto_mask = df_plot[label_col].isin(pareto_labels)
    
    # Separate highlighted model if specified
    highlight_data = None
    if highlight_model is not None:
        highlight_mask = df_plot[label_col] == highlight_model
        if highlight_mask.any():
            highlight_data = df_plot[highlight_mask].iloc[0]
            df_plot = df_plot[~highlight_mask]
            # Recompute pareto_mask after removing highlight
            pareto_mask = df_plot[label_col].isin(pareto_labels)
    
    non_pareto = df_plot[~pareto_mask]
    pareto_in_view = df_plot[pareto_mask]
    if y_label.strip().lower() == 'rtfx':
        non_pareto = non_pareto[non_pareto.apply(lambda row: marker_within_bounds(row[x_col], row[y_col] * y_fact), axis=1)]
        pareto_in_view = pareto_in_view[pareto_in_view.apply(lambda row: marker_within_bounds(row[x_col], row[y_col] * y_fact * y_offset), axis=1)]

    # Plot non-Pareto points
    plt.scatter(
        non_pareto[x_col],
        non_pareto[y_col] * y_fact,
        s=120,
        alpha=0.6,
        color='steelblue',
        edgecolors='navy',
        linewidths=1.5,
        label='Other models',
        zorder=2
    )

    # For RTFx plot, offset Pareto points and line for visibility
    y_offset = 1.1

    # Plot Pareto points with different style (offset vertically if RTFx)
    plt.scatter(
        pareto_in_view[x_col],
        pareto_in_view[y_col] * y_fact * y_offset,
        s=180,
        alpha=0.9,
        color='crimson',
        edgecolors='darkred',
        linewidths=2,
        label='Pareto frontier',
        zorder=3,
        marker='D'
    )

    # Pareto frontier line (offset vertically if RTFx)
    plt.plot(
        pareto_df[x_col],
        pareto_df[y_col] * y_fact * y_offset,
        linewidth=3,
        color="crimson",
        alpha=0.7,
        linestyle='--',
        zorder=1
    )

    # Plot highlighted model with star marker if specified
    if highlight_data is not None:
        h_yval = highlight_data[y_col] * y_fact * (y_offset if y_label.strip().lower() == 'rtfx' else 1.0)
        plt.scatter(
            highlight_data[x_col],
            h_yval,
            s=300,
            alpha=1.0,
            color='gold',
            edgecolors='darkorange',
            linewidths=3,
            marker='*',
            zorder=10
        )

    if y_log_scale:
        plt.yscale("log")

    plt.grid(True, alpha=0.3, linewidth=1)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    # Label all points with improved styling
    # Margin for label cutoff (as fraction of axis range)
    margin_x = 0.07 if x_lim is not None and y_label.strip().lower() == 'rtfx' else (0.02 if x_lim is not None else 0.0)
    margin_y = 0.02 if y_lim is not None else 0.0
    x_max = x_lim[1] if x_lim is not None else df_plot[x_col].max()
    y_max = y_lim[1] if y_lim is not None else (df_plot[y_col] * y_fact).max()

    def label_within_bounds(x, y):
        return (x < x_max * (1 - margin_x)) and (y < y_max * (1 - margin_y))

    # First plot non-Pareto labels (lower zorder)
    # Sort by WER descending so lower WER models are plotted last (on top)
    non_pareto_sorted = non_pareto.sort_values(by=x_col, ascending=False)
    for _, row in non_pareto_sorted.iterrows():
        xval = row[x_col]
        yval = row[y_col] * y_fact
        if y_label.strip().lower() == 'rtfx' and not label_within_bounds(xval, yval):
            continue
        plt.text(
            xval,
            yval,
            row[label_col],
            fontsize=14,
            fontweight='normal',
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='lightyellow',
                edgecolor='gray',
                alpha=0.7,
                linewidth=0.5
            ),
            zorder=4
        )

    # Then plot Pareto labels on top (higher zorder)
    # Sort by WER descending so lower WER models are plotted last (on top)
    pareto_sorted = pareto_in_view.sort_values(by=x_col, ascending=False)
    for _, row in pareto_sorted.iterrows():
        xval = row[x_col]
        yval = row[y_col] * y_fact * (y_offset if y_label.strip().lower() == 'rtfx' else 1.0)
        if y_label.strip().lower() == 'rtfx' and not label_within_bounds(xval, yval):
            continue
        plt.text(
            xval,
            yval,
            row[label_col],
            fontsize=16,
            fontweight='bold',
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='white',
                edgecolor='darkred',
                alpha=0.9,
                linewidth=1.5
            ),
            zorder=5
        )

    # Plot highlighted model label to the right of the star
    if highlight_data is not None:
        xval = highlight_data[x_col]
        yval = highlight_data[y_col] * y_fact * (y_offset if y_label.strip().lower() == 'rtfx' else 1.0)
        # Offset x position to the right (percentage of x range)
        x_range = (x_lim[1] - x_lim[0]) if x_lim is not None else (df_plot[x_col].max() - df_plot[x_col].min())
        label_xval = xval + (x_range * 0.03)  # 3% of x-axis range to the right
        plt.text(
            label_xval,
            yval,
            highlight_data[label_col],
            fontsize=18,
            fontweight='bold',
            ha="left",
            va="center",
            color='red',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='red',
                alpha=0.95,
                linewidth=2.5
            ),
            zorder=100
        )

    plt.xlabel(x_label, fontsize=font_size, fontweight='bold')
    plt.ylabel(y_label, fontsize=font_size, fontweight='bold')
    if title:
        plt.title(title, fontsize=font_size + 2, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='both', which='minor', labelsize=font_size)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    plt.style.use('default')  # Reset style
