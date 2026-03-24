import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_stage_progression_and_colors(df_plot):
    stage_progression = df_plot.groupby('simplified_stage')['task_progression'].mean().sort_values()
    ordered_labels = stage_progression.index.tolist()

    unique_tps = sorted(stage_progression.unique())
    success_tp = stage_progression.get('Success', 1.0)
    failure_tps = [tp for tp in unique_tps if tp != success_tp]

    cmap = plt.get_cmap('Reds')
    tp_to_color = {}
    num_failure_tps = len(failure_tps)
    for i, tp in enumerate(failure_tps):
        tp_to_color[tp] = cmap((i + 1) / (num_failure_tps + 1))

    color_dict = {'Success': 'lightgreen'}
    for stage in ordered_labels:
        if stage == 'Success':
            continue
        tp = stage_progression[stage]
        color_dict[stage] = tp_to_color.get(tp, 'darkred')

    return stage_progression, ordered_labels, color_dict


def plot_stage_frequency(df, ax, model_to_marker=None, model_to_color=None, title=None):
    if df is None or df.empty or 'stage' not in df.columns or 'task_progression' not in df.columns:
        return

    df_plot = df.copy()
    df_plot['simplified_stage'] = df_plot['stage'].apply(
        lambda x: 'Success' if str(x).lower() == 'success' else str(x).split('_')[0].capitalize()
    )

    stage_progression, ordered_labels, color_dict = get_stage_progression_and_colors(df_plot)

    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']

    x_offset_width = 0.8 / len(unique_models)
    x_base = np.arange(len(ordered_labels))
    for i, model in enumerate(unique_models):
        model_df = df_plot[df_plot['model'] == model]
        stage_counts = model_df['simplified_stage'].value_counts()
        total_rows = len(model_df)
        proportions = [stage_counts.get(l, 0) / total_rows if total_rows > 0 else 0 for l in ordered_labels]

        x_pos = x_base + (i - (len(unique_models)-1)/2) * x_offset_width
        colors = [color_dict[label] for label in ordered_labels]

        ax.bar(x_pos, proportions, width=x_offset_width, color=colors, edgecolor='none', alpha=0.8)

        if model_to_marker and model in model_to_marker:
            marker = model_to_marker[model]
            for j, p in enumerate(proportions):
                if p > 0:
                    ax.scatter(x_pos[j], p + 0.02, marker=marker, color='black', edgecolor='black', s=30, zorder=5)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Frequency (Proportion)")
    ax.set_xticks(x_base)
    ax.set_xticklabels(ordered_labels, rotation=45, ha='right')
    if title:
        ax.set_title(title)


def plot_stage_frequency_per_task(df, ax, model_to_marker=None, model_to_color=None, title=None):
    if df is None or df.empty or 'stage' not in df.columns or 'task' not in df.columns or 'task_progression' not in df.columns:
        return

    if 'perturbation' in df.columns:
        df_filtered = df[~df['perturbation'].astype(str).str.contains('SB-VRB')].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return

    df_plot = df_filtered.copy()
    df_plot['simplified_stage'] = df_plot['stage'].apply(
        lambda x: 'Success' if str(x).lower() == 'success' else str(x).split('_')[0].capitalize()
    )

    stage_progression, ordered_stages, color_dict = get_stage_progression_and_colors(df_plot)

    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']
    unique_tasks = sorted(df_plot['task'].unique())

    x_offset_width = 0.8 / len(unique_models)
    x_base = np.arange(len(unique_tasks))

    from matplotlib.patches import Patch
    stage_legend_elements = [Patch(facecolor=color_dict[stage], label=stage) for stage in ordered_stages]

    for i, model in enumerate(unique_models):
        model_df = df_plot[df_plot['model'] == model]
        if model_df.empty:
            continue

        ct = pd.crosstab(model_df['task'], model_df['simplified_stage'], normalize='index')

        for task in unique_tasks:
            if task not in ct.index:
                ct.loc[task] = 0.0
        for stage in ordered_stages:
            if stage not in ct.columns:
                ct[stage] = 0.0

        ct = ct.reindex(index=unique_tasks, columns=ordered_stages).fillna(0)

        x_pos = x_base + (i - (len(unique_models)-1)/2) * x_offset_width
        bottom = np.zeros(len(unique_tasks))

        for stage in ordered_stages:
            proportions = ct[stage].values
            ax.bar(x_pos, proportions, width=x_offset_width, bottom=bottom,
                   color=color_dict[stage], edgecolor='none', alpha=0.8)
            bottom += proportions

        if model_to_marker and model in model_to_marker:
            marker = model_to_marker[model]
            for j in range(len(unique_tasks)):
                if bottom[j] > 0:
                    ax.scatter(x_pos[j], bottom[j] + 0.02, marker=marker, color='black', edgecolor='black', s=30, zorder=5)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Frequency")
    ax.set_xticks(x_base)
    ax.set_xticklabels(unique_tasks, rotation=45, ha='right')

    ax.legend(handles=stage_legend_elements, title='Stage', loc='upper center',
              bbox_to_anchor=(0.5, -0.6), ncol=min(len(ordered_stages), 5))

    if title:
        ax.set_title(title)

    ax.get_figure().subplots_adjust(bottom=0.35)
