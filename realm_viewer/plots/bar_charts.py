import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols=None, model_names=None, model_colors=None, title=None, ylabel=None, fontsize=12, hline=None, hline_label=None):
    x_positions = np.arange(len(labels))

    group_names = [l.split(' - ')[0] for l in labels]
    unique_groups = []
    unique_indices = []
    for i, g in enumerate(group_names):
        if g not in unique_groups:
            unique_groups.append(g)
            unique_indices.append(i)

    ax.bar(x_positions, values, color=colors, edgecolor='none')

    if hline is not None:
        ax.axhline(y=hline, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label=hline_label)
        if hline_label:
            ax.legend(loc='upper right', fontsize=fontsize-4)

    if symbols:
        for i, (v, s) in enumerate(zip(values, symbols)):
            if s:
                ax.scatter(x_positions[i], v + 0.05, marker=s, color='black', edgecolor='black', s=50, zorder=5)

    for i, group_label in enumerate(unique_groups):
        start_idx = unique_indices[i]
        end_idx = unique_indices[i+1] if i + 1 < len(unique_indices) else len(labels)
        group_size = end_idx - start_idx

        if i > 0:
            ax.axvline(x_positions[start_idx] - 0.5, color="gray", linewidth=0.8, linestyle='--')

        center_x = start_idx + (group_size - 1) / 2
        ax.text(center_x, -0.05, group_label, ha='right', va="top", fontsize=fontsize-2, rotation=45)

    ax.set_xlim(-0.5, len(labels) - 0.5)
    max_val = max(values) if values else 1.0
    if hline is not None:
        max_val = max(max_val, hline)
    ax.set_ylim(0, max(1.1, max_val + 0.15))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=fontsize+2)

    ax.get_figure().subplots_adjust(bottom=0.35)
    return ax


def render_metric_bar_chart(df, metric_col, title, ylabel, model_to_marker, model_to_color, color=None, hline=None, hline_label=None):
    """Helper to render a grouped bar chart for a specific metric."""
    if metric_col not in df.columns:
        st.info(f"Column '{metric_col}' not found in data.")
        return

    try:
        metric_df = df[df[metric_col].notna()].copy()
        if metric_df.empty:
            st.info(f"No data available for metric '{metric_col}'.")
            return

        stats = metric_df.groupby(['task', 'model'])[metric_col].mean().reset_index()
        stats = stats.sort_values(['task', 'model'])

        labels = [f"{row['task']} - {row['model']}" for _, row in stats.iterrows()]
        values = stats[metric_col].tolist()

        if model_to_color:
            colors = [model_to_color[row['model']] for _, row in stats.iterrows()]
        else:
            colors = [color if color else 'gold'] * len(labels)

        symbols = [model_to_marker[row['model']] for _, row in stats.iterrows()]
        model_names = stats['model'].tolist()

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel=ylabel, title=title, hline=hline, hline_label=hline_label)

        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error plotting {title}: {e}")
