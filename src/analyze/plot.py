"""
Plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = ("HR", "NDCG", "MRR")
DEFAULT_CORR_TYPES = ("pearson", "spearman", "kendall")


def plot_metrics_vs_corrtype_vs_cutoffs(split_corr_dfs, figsize=(12, 9),
                                        metrics=DEFAULT_METRICS,
                                        corr_types=DEFAULT_CORR_TYPES):

    fig, axes = plt.subplots(nrows=len(metrics), ncols=len(corr_types), figsize=figsize)

    for i, metric in enumerate(metrics):
        for j, corr_type in enumerate(corr_types):
            ax = axes[i, j]
            for split_name, df in split_corr_dfs.items():
                cutoffs, values = get_metric_data(df, metric, corr_type)
                # Create equally spaced x positions
                x_positions = list(range(len(cutoffs)))
                marker = get_marker(split_name)
                ax.plot(x_positions, values, marker=marker, ms=6, label=split_name)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(cutoffs)
            ax.set_title(f"{metric} - {corr_type.title()}")
            ax.set_xlabel("Cutoff (K)")
            ax.set_ylabel("Correlation")
            ax.grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6,
               fontsize=8, framealpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    return fig


def plot_avg_metrics_across_metrics_vs_cutoffs(split_corr_dfs, figsize=(15, 4),
                                               metrics=DEFAULT_METRICS,
                                               corr_types=DEFAULT_CORR_TYPES):

    fig, axes = plt.subplots(nrows=1, ncols=len(corr_types), figsize=figsize)

    for j, corr_type in enumerate(corr_types):
        ax = axes[j]
        for split_name, df in split_corr_dfs.items():
            cutoffs, _ = get_metric_data(df, "HR", corr_type)
            x_positions = list(range(len(cutoffs)))
            avg_values = []
            for idx in range(len(cutoffs)):
                vals = []
                for metric in metrics:
                    _, values = get_metric_data(df, metric, corr_type)
                    vals.append(values[idx])
                avg_values.append(np.mean(vals))
            marker = get_marker(split_name)
            ax.plot(x_positions, avg_values, marker=marker, ms=6.5, label=split_name)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(cutoffs)
        ax.set_title(corr_type.title())
        ax.set_xlabel("Cutoff (K)")
        ax.set_ylabel("Avg. Correlation")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=7,
               fontsize=8, framealpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.87])

    return fig


def plot_avg_cutoffs_across_cutoffs_vs_corrtype(split_corr_dfs, figsize=(15, 4),
                                                metrics=DEFAULT_METRICS,
                                                corr_types=DEFAULT_CORR_TYPES):

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=figsize)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for split_name, df in split_corr_dfs.items():
            avg_corrs = []
            for corr_type in corr_types:
                avg_corr = get_avg_metric_across_cutoffs(df, metric, corr_type)
                avg_corrs.append(avg_corr)
            marker = get_marker(split_name)
            ax.plot(corr_types, avg_corrs, marker=marker, label=split_name)
        ax.set_title(metric)
        ax.set_xlabel("Correlation Type")
        ax.set_ylabel("Avg. Correlation (over cutoffs)")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=7,
               fontsize=8, framealpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.87])

    return fig


def plot_specific_cutoff_corr_vs_corrtype(split_corr_dfs, cutoff="@10", figsize=(15, 4),
                                          metrics=DEFAULT_METRICS, corr_types=DEFAULT_CORR_TYPES):

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=figsize)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for split_name, df in split_corr_dfs.items():
            vals = []

            for corr_type in corr_types:
                cutoffs, values = get_metric_data(df, metric, corr_type)
                idx = None
                try:
                    idx = [str(c) for c in cutoffs].index(cutoff.strip("@"))
                except ValueError:
                    for k, c in enumerate(cutoffs):
                        if np.isclose(c, float(cutoff.strip("@"))):
                            idx = k
                            break
                if idx is not None:
                    vals.append(values[idx])
                else:
                    vals.append(np.nan)
            marker = get_marker(split_name)
            ax.plot(corr_types, vals, marker=marker, label=split_name)
        ax.set_title(f"{metric} at {cutoff}")
        ax.set_xlabel("Correlation Type")
        ax.set_ylabel("Correlation")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=7,
               fontsize=8, framealpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.87])

    return fig


def plot_metric_vs_cutoffs_by_split_pair(split_corr_dfs, figsize=(18, 9),
                                         metrics=DEFAULT_METRICS,
                                         corr_types=DEFAULT_CORR_TYPES):

    n_split = len(split_corr_dfs)

    fig, axes = plt.subplots(nrows=len(metrics), ncols=n_split, figsize=figsize)

    for col, (split_name, df) in enumerate(split_corr_dfs.items()):
        marker = get_marker(split_name)
        for row, metric in enumerate(metrics):
            ax = axes[row, col] if n_split > 1 else axes[row]
            for corr_type in corr_types:
                cutoffs, values = get_metric_data(df, metric, corr_type)
                x_positions = list(range(len(cutoffs)))
                ax.plot(x_positions, values, marker=marker, label=corr_type.title())
                ax.set_xticks(x_positions)
                ax.set_xticklabels(cutoffs)
            if col == 0:
                ax.set_title(f"{metric} - {split_name}", fontsize=10)
            else:
                ax.set_title(f"{split_name}", fontsize=10)
            ax.set_xlabel("Cutoff (K)")
            ax.set_ylabel("Correlation")
            ax.grid(True)
            if row == 0:
                ax.legend(fontsize=8, framealpha=0.5)
    plt.tight_layout()

    return fig


def get_marker(split_name):

    if 'LLO' in split_name:
        marker = "v"
    elif 'Succ' in split_name:
        marker = "s"
    else:
        marker = "o"

    return marker


def plot_corr_reference_vs_others(data_dict, split_types, ref_quant, other_quants, metric, cutoff,
                                  corr_types=("pearson", "spearman", "kendall"), figsize=(18,5)):
    fig, axes = plt.subplots(1, len(corr_types), figsize=figsize, sharey=True)
    for i, ct in enumerate(corr_types):
        ax = axes[i]
        for sp in split_types:
            y_vals = []
            for q in other_quants:
                key_ref = f"{ref_quant}_{sp}"
                key_other = f"{q}_{sp}"
                df_ref = data_dict.get(key_ref)
                df_other = data_dict.get(key_other)
                if df_ref is None or df_other is None:
                    y_vals.append(np.nan)
                else:
                    corr_vals = get_pairwise_corr_at_cutoff(df_ref, df_other, metric, cutoff)
                    y_vals.append(corr_vals.get(ct, np.nan))
            ax.plot(other_quants, y_vals, marker='o', label=sp)
        ax.set_title(f"{ct.title()}")
        # ax.set_xlabel("q")
        if i == 0:
            ax.set_ylabel("Correlation")
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.suptitle(f"{metric}@{cutoff} for correlations with {ref_quant}")
    plt.tight_layout(rect=[0, 0, 1, 1])
    return fig


def get_pairwise_corr_at_cutoff(df_ref, df_other, metric, cutoff_val):
    col_name = f"{metric}@{cutoff_val}"
    if col_name not in df_ref.columns or col_name not in df_other.columns:
        return {'pearson': np.nan, 'spearman': np.nan, 'kendall': np.nan}
    df_merge = pd.merge(
        df_ref[['grid_point', col_name]],
        df_other[['grid_point', col_name]],
        on='grid_point',
        suffixes=("_ref", "_other")
    ).dropna()
    if df_merge.empty:
        return {'pearson': np.nan, 'spearman': np.nan, 'kendall': np.nan}
    x = df_merge[f"{col_name}_ref"].values
    y = df_merge[f"{col_name}_other"].values
    return {
        'pearson': np.corrcoef(x, y)[0, 1],
        'spearman': pd.Series(x).corr(pd.Series(y), method='spearman'),
        'kendall': pd.Series(x).corr(pd.Series(y), method='kendall')
    }


def get_metric_data(df, metric, corr_type):

    # Select the row for the given correlation type (assumed to be lowercase)
    row = df.loc[corr_type.lower()]
    # Filter columns that match the metric name followed by "@"
    metric_cols = [col for col in row.index if col.startswith(metric + "@")]

    def get_cutoff(col):
        try:
            return float(col.split("@")[1])
        except Exception:
            return np.inf

    # Sort by numeric cutoff value
    sorted_pairs = sorted(((get_cutoff(col), col) for col in metric_cols), key=lambda x: x[0])
    cutoffs = [pair[0] for pair in sorted_pairs]
    values = [row[pair[1]] for pair in sorted_pairs]

    return cutoffs, values


def get_avg_metric_across_cutoffs(df, metric, corr_type):

    _, values = get_metric_data(df, metric, corr_type)
    return np.mean(values)
