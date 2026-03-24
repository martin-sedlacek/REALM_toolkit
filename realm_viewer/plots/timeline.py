import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_task_progression_timesteps_per_task(df, ax, model_to_marker=None, model_to_color=None, title=None):
    col_name = None
    for col in ['task_progression_timesteps', 'task_progression_timestamps', 'task_progression_timstamps']:
        if col in df.columns:
            col_name = col
            break

    if col_name is None or 'task' not in df.columns:
        return

    if 'perturbation' in df.columns:
        df_filtered = df[~df['perturbation'].astype(str).str.contains('SB-VRB')].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return

    df_plot = df_filtered.copy()

    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']
    unique_tasks = sorted(df_plot['task'].unique())

    try:
        df_plot['parsed_ts'] = df_plot[col_name].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        max_stages = df_plot['parsed_ts'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
        if max_stages == 0:
            return

        def get_durations(ts_list):
            if not isinstance(ts_list, list) or len(ts_list) == 0:
                return []
            durations = []
            prev_ts = 0
            for ts in ts_list:
                durations.append(ts - prev_ts)
                prev_ts = ts
            return durations

        x_offset_width = 0.8 / len(unique_models)
        x_base = np.arange(len(unique_tasks))

        cmap = plt.get_cmap('viridis')
        stage_colors = [cmap(i / max_stages) for i in range(max_stages)]

        for i, model in enumerate(unique_models):
            model_df = df_plot[df_plot['model'] == model].copy()
            if model_df.empty:
                continue

            model_df['durations'] = model_df['parsed_ts'].apply(get_durations)
            model_df['durations'] = model_df['durations'].apply(lambda x: x + [np.nan] * (max_stages - len(x)))
            durations_df = pd.DataFrame(model_df['durations'].tolist(), index=model_df.index)
            durations_df['task'] = model_df['task']

            mean_durations = durations_df.groupby('task').mean().reindex(unique_tasks).fillna(0)

            x_pos = x_base + (i - (len(unique_models)-1)/2) * x_offset_width
            bottom = np.zeros(len(unique_tasks))

            for stage_idx in range(max_stages):
                vals = mean_durations[stage_idx].values
                ax.bar(x_pos, vals, width=x_offset_width, bottom=bottom,
                       color=stage_colors[stage_idx], edgecolor='none', alpha=0.8)
                bottom += vals

            if model_to_marker and model in model_to_marker:
                marker = model_to_marker[model]
                for j in range(len(unique_tasks)):
                    if bottom[j] > 0:
                        ax.scatter(x_pos[j], bottom[j] + 5, marker=marker, color='black', edgecolor='black', s=30, zorder=5)

        ax.set_ylabel("Average Timesteps")
        ax.set_xticks(x_base)
        ax.set_xticklabels(unique_tasks, rotation=45, ha='right')

        from matplotlib.patches import Patch
        stage_legend_elements = [Patch(facecolor=stage_colors[i], label=f'Stage {i+1}') for i in range(max_stages)]
        ax.legend(handles=stage_legend_elements, title='Stage Step', loc='upper center',
                  bbox_to_anchor=(0.5, -0.6), ncol=min(max_stages, 5))

        if title:
            ax.set_title(title)

        ax.get_figure().subplots_adjust(bottom=0.35)

    except Exception as e:
        print(f"Error plotting average timesteps: {e}")
