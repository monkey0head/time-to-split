"""
Compute correlations between metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_corr_df(df1, df2, split_names=("A", "B"), metrics=None):

    corr_summary = {}
    if metrics is None:
        common_metrics = find_common_metrics(df1, df2)
    else:
        common_metrics = metrics

    for metric in common_metrics:
        common_cutoffs = find_common_cutoffs(df1, df2, metric)

        for cutoff in common_cutoffs:
            col_name = f"{metric}@{cutoff}"
            df_merge = pd.merge(
                df1[['grid_point', col_name]],
                df2[['grid_point', col_name]],
                on='grid_point',
                suffixes=(f'_{split_names[0]}', f'_{split_names[1]}')
            ).dropna()

            if df_merge.empty:
                corr_summary[col_name] = {'pearson': np.nan, 'spearman': np.nan, 'kendall': np.nan}
                continue

            x = df_merge[f"{col_name}_{split_names[0]}"].values
            y = df_merge[f"{col_name}_{split_names[1]}"].values
            corr_summary[col_name] = {
                'pearson': np.corrcoef(x, y)[0, 1],
                'spearman': pd.Series(x).corr(pd.Series(y), method='spearman'),
                'kendall': pd.Series(x).corr(pd.Series(y), method='kendall')
            }

    corr_df = pd.DataFrame(corr_summary, index=['pearson', 'spearman', 'kendall'])

    return corr_df


def plot_scatterplots(df1, df2, split_names):

    common_metrics = find_common_metrics(df1, df2)
    figures = []

    for metric in common_metrics:
        fig = plot_scatterplots_for_metric(df1, df2, split_names, metric)
        figures.append(fig)

    return figures


def plot_scatterplots_for_metric(df1, df2, split_names, metric, color_by='model'):
    """
    Plot scatterplots comparing metrics between two dataframes, with optional coloring.
    
    Args:
        df1: First dataframe with results
        df2: Second dataframe with results
        split_names: Names for the two splits being compared
        metric: Metric to compare (e.g., 'NDCG', 'HR')
        color_by: Optional column name to use for marker colors. If None or column doesn't exist,
            markers will be the same color.
    """

    common_cutoffs = find_common_cutoffs(df1, df2, metric)
    if not common_cutoffs:
        return

    fig, axes = plt.subplots(1, len(common_cutoffs), squeeze=False,
                             figsize=(3 * len(common_cutoffs), 4))

    fig.suptitle(f"{metric}: {split_names[0]} vs {split_names[1]}", fontsize=14, y=0.95)

    # Check if color_by column exists in both dataframes
    use_coloring = False
    if color_by is not None and color_by in df1.columns and color_by in df2.columns:
        use_coloring = True

    for i, cutoff in enumerate(common_cutoffs):
        col_name = f"{metric}@{cutoff}"

        # Include color_by column in merge if we're using coloring
        merge_cols = ['grid_point', col_name]
        if use_coloring:
            merge_cols.append(color_by)

        df_merge = pd.merge(
            df1[merge_cols],
            df2[merge_cols],
            on='grid_point',
            suffixes=(f'_{split_names[0]}', f'_{split_names[1]}')
        ).dropna()

        ax = axes[0, i]
        if df_merge.empty:
            ax.set_title(f"@{cutoff}\nNo Data")
            continue

        x = df_merge[f"{col_name}_{split_names[0]}"].values
        y = df_merge[f"{col_name}_{split_names[1]}"].values
        # Get color values if using coloring
        hue = df_merge[f"{color_by}_{split_names[0]}"] if use_coloring else None

        # Calculate correlation coefficients
        pearson = np.corrcoef(x, y)[0, 1]
        spearman = pd.Series(x).corr(pd.Series(y), method='spearman')
        kendall = pd.Series(x).corr(pd.Series(y), method='kendall')

        # Create scatter plot with or without coloring
        if use_coloring:
            legend = True if i == len(common_cutoffs) - 1 else False
            sns.scatterplot(x=x, y=y, ax=ax, s=50, hue=hue, legend=legend)
            if legend:
                ax.legend_.remove()
        else:
            sns.scatterplot(x=x, y=y, ax=ax, s=50)

        # Add regression line
        slope, intercept = np.polyfit(x, y, 1)
        x_vals = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_vals, intercept + slope * x_vals, color='red')

        ax.set_title(f"{metric}@{cutoff} \n Pearson: {pearson:.3f} \n "
                     f"Spearman: {spearman:.3f} \n Kendall: {kendall:.3f}")
        ax.set_xlabel(f"{split_names[0]}")
        if i == 0:
            ax.set_ylabel(f"{split_names[1]}")
        else:
            ax.set_ylabel("")

    # Add single legend if coloring
    if use_coloring:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title=color_by, bbox_to_anchor=(1.07, 0.5))

    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    return fig


def plot_heatmaps(df1, df2, split_names, methods=('pearson', 'spearman', 'kendall')):

    common_metrics = find_common_metrics(df1, df2)
    n_metrics = len(common_metrics)

    fig, axes = plt.subplots(len(methods), n_metrics,
                             figsize=(4 * n_metrics, 4 * len(methods)))
    fig.suptitle(f"Heatmaps: {split_names[0]} (A) vs {split_names[1]} (B)", fontsize=15, y=0.95)

    for j, metric in enumerate(common_metrics):
        cols1 = sorted([col for col in df1.columns if col.startswith(f"{metric}@")],
                       key=lambda x: int(x.split('@')[1]))
        cols2 = sorted([col for col in df2.columns if col.startswith(f"{metric}@")],
                       key=lambda x: int(x.split('@')[1]), reverse=True)
        df1_sub = df1[['grid_point'] + cols1].copy()
        df2_sub = df2[['grid_point'] + cols2].copy()

        df1_sub = df1_sub.rename(
            columns=lambda x: f"A_@{x.split('@')[1]}" if x != "grid_point" else x)
        df2_sub = df2_sub.rename(
            columns=lambda x: f"B_@{x.split('@')[1]}" if x != "grid_point" else x)
        df_merge = pd.merge(df1_sub, df2_sub, on="grid_point", how="inner").set_index("grid_point")
        new_order = [col for col in df_merge.columns if col.startswith("A_")]
        new_order += [col for col in df_merge.columns if col.startswith("B_")]
        df_merge = df_merge[new_order]

        for i, method in enumerate(methods):
            ax = axes[i, j] if n_metrics > 1 else axes[i]
            corr_matrix = df_merge.corr(method=method)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax,
                        annot_kws={"size": 8}, fmt=".2f", cbar=False)
            if j == 0:
                ax.tick_params(axis='y', labelleft=True, labelsize=8)
                ax.set_ylabel(method, fontsize=12)
            else:
                ax.tick_params(axis='y', labelleft=False)
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelsize=8)
            if i == 0:
                ax.set_title(metric, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    return fig


def find_common_metrics(df1, df2, metric_order=('NDCG', 'HR', 'MRR', 'COV')):

    metrics1 = {col.split('@')[0] for col in df1.columns if col != 'grid_point'}
    metrics2 = {col.split('@')[0] for col in df2.columns if col != 'grid_point'}

    common_metrics = [metric for metric in metric_order
                      if metric in metrics1 and metric in metrics2]

    return common_metrics


def find_common_cutoffs(df1, df2, metric):

    cols1 = [col for col in df1.columns if col.startswith(f"{metric}@")]
    cols2 = [col for col in df2.columns if col.startswith(f"{metric}@")]
    cutoffs1 = {col.split('@')[1] for col in cols1}
    cutoffs2 = {col.split('@')[1] for col in cols2}

    common_cutoffs = sorted(list(cutoffs1 & cutoffs2), key=lambda x: int(x))

    return common_cutoffs