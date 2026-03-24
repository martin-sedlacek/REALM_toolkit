import numpy as np
from scipy.stats import beta


def plot_bayesian_violin(ax, labels, successes, failures, colors, symbols=None, model_names=None, model_colors=None, title=None, fontsize=20):
    x_positions = np.arange(len(labels))
    successes_arr = np.array(successes, dtype=float).flatten()
    failures_arr = np.array(failures, dtype=float).flatten()
    colors_arr = np.array(colors).flatten()
    symbols_arr = np.array(symbols).flatten() if symbols is not None else None

    task_names = [l.split(' - ')[0] for l in labels]
    unique_tasks = []
    unique_indices = []

    for data_idx, (S_raw, F_raw, color) in enumerate(zip(successes_arr, failures_arr, colors_arr)):
        S = int(S_raw) if not np.isnan(S_raw) else None
        F = int(F_raw) if not np.isnan(F_raw) else None

        if S is None or F is None or (S + F) == 0:
            continue

        task_name = task_names[data_idx]
        if task_name not in unique_tasks:
            unique_tasks.append(task_name)
            unique_indices.append(data_idx)

        alpha = S + 1
        beta_param = F + 1

        y = np.linspace(0.001, 0.999, 300)
        density = beta.pdf(y, alpha, beta_param)

        if density.max() == 0:
            continue

        mask = density >= (density.max() * 0.0001)
        y_clipped = y[mask]
        density_clipped = density[mask]

        x = density_clipped / density.max() * 0.325

        ax.fill_betweenx(y_clipped, x_positions[data_idx] - x, x_positions[data_idx] + x, color=color, alpha=1)

        posterior_mean = alpha / (alpha + beta_param)
        ax.hlines(y=posterior_mean, xmin=x_positions[data_idx] - 0.105, xmax=x_positions[data_idx] + 0.105, color='black', linewidth=2)

        if symbols_arr is not None and data_idx < len(symbols_arr):
            marker = symbols_arr[data_idx]
            if marker:
                y_pos = y_clipped.max() + 0.05 if len(y_clipped) > 0 else 1.05
                ax.scatter(x_positions[data_idx], y_pos, marker=marker, color='black', edgecolor='black', s=50, zorder=5)

    for i, task_label in enumerate(unique_tasks):
        start_idx = unique_indices[i]
        end_idx = unique_indices[i+1] if i + 1 < len(unique_indices) else len(labels)
        group_size = end_idx - start_idx

        if i > 0:
            ax.axvline(x_positions[start_idx] - 0.5, color="gray", linewidth=0.8, linestyle='--')

        center_x = start_idx + (group_size - 1) / 2
        ax.text(center_x, -0.05, task_label, ha='right', va="top", fontsize=fontsize-2, rotation=45)

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Success Rate", fontsize=fontsize, labelpad=15)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=fontsize+2, y=1.05, fontstyle='italic')
    else:
        ax.set_title("Binary Success Rate (Bayesian Posterior)", fontsize=fontsize+2, y=1.05, fontstyle='italic')

    ax.get_figure().subplots_adjust(bottom=0.35)
    return ax
