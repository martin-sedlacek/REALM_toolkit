import streamlit as st
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import dashboard_utils
from collections import defaultdict
import pandas as pd
import ast
from scipy.stats import beta

st.set_page_config(layout="wide", page_title="Experiment Dashboard")

MARKERS = ['o', 's', '^', 'v', '<', '>', 'D', 'P', 'X', '*']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# --- Custom CSS for tree-style sidebar buttons ---
st.markdown("""
<style>
section[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.1rem 0.4rem !important;
    font-size: 0.875rem !important;
    width: 100% !important;
    color: inherit !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    box-shadow: none !important;
    min-height: 0 !important;
    line-height: 1.6 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(151, 166, 195, 0.15) !important;
}
section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}
section[data-testid="stSidebar"] .stButton {
    margin-bottom: -0.6rem !important;
}
</style>
""", unsafe_allow_html=True)

# Define the logs directory
LOGS_DIR = os.environ.get("REALM_LOGS", "logs")
VIDEOS_PER_PAGE = 6

# Initialize session state
if "selected_runs" not in st.session_state:
    st.session_state.selected_runs = []
if "expanded_folders" not in st.session_state:
    st.session_state.expanded_folders = set()
if "video_page" not in st.session_state:
    st.session_state.video_page = 0
if "last_selected_runs" not in st.session_state:
    st.session_state.last_selected_runs = []
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# --- Caching Wrapper Functions ---
@st.cache_data(ttl=60)
def get_cached_subdirectories(path):
    return dashboard_utils.get_subdirectories(path)

@st.cache_data(ttl=60)
def get_cached_is_experiment_folder(path):
    return dashboard_utils.is_experiment_folder(path)

@st.cache_data(ttl=10)
def get_cached_reports(path):
    return dashboard_utils.load_reports(path)

@st.cache_data(ttl=10)
def get_cached_videos(path):
    return dashboard_utils.get_videos(path)

@st.cache_data(ttl=60)
def get_cached_experiment_metadata(path):
    return dashboard_utils.load_experiment_metadata(path)

# --- Tree callbacks ---
def _toggle_folder(path):
    if path in st.session_state.expanded_folders:
        st.session_state.expanded_folders.discard(path)
    else:
        st.session_state.expanded_folders.add(path)

def has_experiments_recursively(base_path, search_query="", match_found_in_path=False):
    """Check if the given path or any of its subdirectories contain an experiment matching the search query."""
    # If a match was already found in the parent path, we just need to know if this branch contains ANY experiment.
    # Otherwise, check if the current directory matches the search query.
    if search_query and not match_found_in_path:
        match_found_in_path = search_query.lower() in os.path.basename(base_path).lower()
    
    # Check if this is an experiment folder
    if get_cached_is_experiment_folder(base_path):
        # If there's a search query, only return true if a match was found in the path or at this folder
        if search_query:
            return match_found_in_path
        return True
    
    # If not an experiment folder, check subdirectories
    subdirs = get_cached_subdirectories(base_path)
    for d in subdirs:
        full_path = os.path.join(base_path, d)
        if has_experiments_recursively(full_path, search_query, match_found_in_path):
            return True
    return False

# --- Tree rendering ---
def render_tree(base_path, depth=0, search_query="", match_found_in_path=False):
    """Render a collapsible directory tree in the sidebar with proper indentation."""
    if depth > 5:
        return

    subdirs = get_cached_subdirectories(base_path)
    if not subdirs:
        return

    for d in subdirs:
        full_path = os.path.join(base_path, d)
        
        # Check if the query matches the current directory
        current_match = match_found_in_path or (search_query.lower() in d.lower() if search_query else False)
        
        # Check if this branch has any experiments matching the query (or if a parent matched)
        if not has_experiments_recursively(full_path, search_query, current_match):
            continue
            
        is_exp = get_cached_is_experiment_folder(full_path)
        sub_subdirs = get_cached_subdirectories(full_path)
        
        # Do not treat experiment folders as having children in the tree to prevent them from rolling down
        has_children = len(sub_subdirs) > 0 and not is_exp
        
        # Expand automatically if there is a search query
        is_expanded = (full_path in st.session_state.expanded_folders) or (search_query != "")
        
        # Build indented label: 2 em-spaces per depth level
        indent = "\u2003" * (depth * 2)

        if has_children:
            chevron = "▾" if is_expanded else "▸"
            label = f"{indent}{chevron} 📁 {d}"
            st.sidebar.button(
                label,
                key=f"tree_{full_path}",
                on_click=_toggle_folder,
                args=(full_path,),
            )
        elif is_exp:
            label = f"{indent}🔬 {d}"
            st.sidebar.checkbox(label, key=f"chk_{full_path}")
        else:
            # Non-experiment leaf folder (should not be reached if has_experiments_recursively is working correctly)
            st.sidebar.markdown(
                f"<div style='padding-left:{depth * 1.5}rem; font-size:0.85rem; color:gray; "
                f"padding-top:0.1rem; padding-bottom:0.1rem;'>\u2003 📁 {d}</div>",
                unsafe_allow_html=True,
            )

        # Recurse into expanded children
        if has_children and is_expanded:
            render_tree(full_path, depth + 1, search_query, current_match)

# --- Video page callbacks ---
def _video_prev_page():
    st.session_state.video_page = max(0, st.session_state.video_page - 1)

def _video_next_page():
    st.session_state.video_page += 1

# --- Video section (fragment for partial reruns) ---
@st.fragment
def render_video_section(video_paths, selected_tasks, selected_perts):
    """Render paginated video gallery. Runs as a fragment so page navigation
    does not trigger a full dashboard rerun."""
    filtered_videos = dashboard_utils.filter_videos(video_paths, selected_tasks, selected_perts)

    if not filtered_videos:
        if video_paths:
            st.info("No videos match the selected filters.")
        else:
            st.info("No videos found.")
        return

    total_videos = len(filtered_videos)
    total_pages = max(1, (total_videos + VIDEOS_PER_PAGE - 1) // VIDEOS_PER_PAGE)

    # Clamp page
    if st.session_state.video_page >= total_pages:
        st.session_state.video_page = total_pages - 1

    start_idx = st.session_state.video_page * VIDEOS_PER_PAGE
    end_idx = min(start_idx + VIDEOS_PER_PAGE, total_videos)

    # Page navigation bar
    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        st.button("← Prev", on_click=_video_prev_page,
                  disabled=(st.session_state.video_page == 0))
    with nav2:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.4rem;'>"
            f"Showing {start_idx + 1}–{end_idx} of {total_videos} videos "
            f"(Page {st.session_state.video_page + 1}/{total_pages})</div>",
            unsafe_allow_html=True,
        )
    with nav3:
        st.button("Next →", on_click=_video_next_page,
                  disabled=(st.session_state.video_page >= total_pages - 1))

    # Render current page
    page_videos = filtered_videos[start_idx:end_idx]
    cols = st.columns(3)
    for i, video_path in enumerate(page_videos):
        with cols[i % 3]:
            st.video(video_path)
            # Use relative path from LOGS_DIR for caption to distinguish run_ids
            st.caption(os.path.relpath(video_path, LOGS_DIR))

def get_stage_progression_and_colors(df_plot):
    # Calculate average task progression for each simplified stage
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

def plot_stage_frequency(df, model_to_marker=None, model_to_color=None):
    if df is None or df.empty or 'stage' not in df.columns or 'task_progression' not in df.columns:
        st.info("No data or required columns ('stage', 'task_progression') found for this plot.")
        return

    # Use original dataframe
    df_plot = df.copy()

    # Simplify stage labels: 'Success' remains 'Success', others take the first word
    df_plot['simplified_stage'] = df_plot['stage'].apply(
        lambda x: 'Success' if str(x).lower() == 'success' else str(x).split('_')[0].capitalize()
    )

    stage_progression, ordered_labels, color_dict = get_stage_progression_and_colors(df_plot)

    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']

    fig, ax = plt.subplots(figsize=(6, 4))

    x_offset_width = 0.8 / len(unique_models)
    x_base = np.arange(len(ordered_labels))    
    for i, model in enumerate(unique_models):
        model_df = df_plot[df_plot['model'] == model]
        stage_counts = model_df['simplified_stage'].value_counts()
        total_rows = len(model_df)
        proportions = [stage_counts.get(l, 0) / total_rows if total_rows > 0 else 0 for l in ordered_labels]
        
        x_pos = x_base + (i - (len(unique_models)-1)/2) * x_offset_width
        colors = [color_dict[label] for label in ordered_labels]
        
        bars = ax.bar(x_pos, proportions, width=x_offset_width, color=colors, edgecolor='none', alpha=0.8)
        
        # Add symbol on top of each group for this model
        if model_to_marker and model in model_to_marker:
            marker = model_to_marker[model]
            m_color = 'black'
            if model_to_color and model in model_to_color:
                m_color = model_to_color[model]
            for j, p in enumerate(proportions):
                if p > 0:
                    ax.scatter(x_pos[j], p + 0.02, marker=marker, color='black', edgecolor='black', s=30, zorder=5)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Frequency (Proportion)")
    ax.set_xticks(x_base)
    ax.set_xticklabels(ordered_labels, rotation=45, ha='right')
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    st.image(buf)

def plot_stage_frequency_per_task(df, model_to_marker=None, model_to_color=None):
    if df is None or df.empty or 'stage' not in df.columns or 'task' not in df.columns or 'task_progression' not in df.columns:
        st.info("No data or required columns ('stage', 'task', 'task_progression') found for this plot.")
        return

    # Filter out SB-VRB perturbation for this specific plot
    if 'perturbation' in df.columns:
        df_filtered = df[~df['perturbation'].astype(str).str.contains('SB-VRB')].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        st.info("No data available after filtering out SB-VRB.")
        return

    df_plot = df_filtered.copy()
    
    # Simplify stage labels: 'Success' remains 'Success', others take the first word
    df_plot['simplified_stage'] = df_plot['stage'].apply(
        lambda x: 'Success' if str(x).lower() == 'success' else str(x).split('_')[0].capitalize()
    )
    
    stage_progression, ordered_stages, color_dict = get_stage_progression_and_colors(df_plot)

    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']
    unique_tasks = sorted(df_plot['task'].unique())

    fig, ax = plt.subplots(figsize=(6, 4))
    
    x_offset_width = 0.8 / len(unique_models)
    x_base = np.arange(len(unique_tasks))
    
    # Track handles for stage legend
    from matplotlib.patches import Patch
    stage_legend_elements = [Patch(facecolor=color_dict[stage], label=stage) for stage in ordered_stages]

    for i, model in enumerate(unique_models):
        model_df = df_plot[df_plot['model'] == model]
        # Create a crosstab for this model: task vs simplified_stage
        if model_df.empty:
            continue
            
        ct = pd.crosstab(model_df['task'], model_df['simplified_stage'], normalize='index')
        
        # Ensure all tasks and stages are represented
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
        
        # Add symbol on top of the stack for this model
        if model_to_marker and model in model_to_marker:
            marker = model_to_marker[model]
            m_color = 'black'
            if model_to_color and model in model_to_color:
                m_color = model_to_color[model]
            for j in range(len(unique_tasks)):
                if bottom[j] > 0:
                    ax.scatter(x_pos[j], bottom[j] + 0.02, marker=marker, color='black', edgecolor='black', s=30, zorder=5)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Frequency")
    ax.set_xticks(x_base)
    ax.set_xticklabels(unique_tasks, rotation=45, ha='right')
    
    # Stage Legend at the bottom
    ax.legend(handles=stage_legend_elements, title='Stage', loc='upper center', 
              bbox_to_anchor=(0.5, -1.0), ncol=min(len(ordered_stages), 5))
    
    buf = io.BytesIO()
    # Manual adjustment to prevent shrinking while making room for labels and legend
    fig.subplots_adjust(bottom=0.35)
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    st.image(buf)

def plot_task_progression_timesteps_per_task(df, model_to_marker=None, model_to_color=None):
    # Find the correct column name for timestamps
    col_name = None
    for col in ['task_progression_timesteps', 'task_progression_timestamps', 'task_progression_timstamps']:
        if col in df.columns:
            col_name = col
            break

    if col_name is None or 'task' not in df.columns:
        st.info("No data or required columns ('task', 'task_progression_timestamps') found for this plot.")
        return

    # Filter out SB-VRB perturbation for this specific plot
    if 'perturbation' in df.columns:
        df_filtered = df[~df['perturbation'].astype(str).str.contains('SB-VRB')].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        st.info("No data available after filtering out SB-VRB.")
        return

    df_plot = df_filtered.copy()
    
    unique_models = sorted(df_plot['model'].unique()) if 'model' in df_plot.columns else ['Unknown']
    unique_tasks = sorted(df_plot['task'].unique())

    try:
        # Parse timestamps into lists
        df_plot['parsed_ts'] = df_plot[col_name].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Determine max stages
        max_stages = df_plot['parsed_ts'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
        if max_stages == 0:
            st.info("No valid timestamp data found.")
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

        fig, ax = plt.subplots(figsize=(6, 4))
        x_offset_width = 0.8 / len(unique_models)
        x_base = np.arange(len(unique_tasks))
        
        cmap = plt.get_cmap('viridis')
        stage_colors = [cmap(i / max_stages) for i in range(max_stages)]

        for i, model in enumerate(unique_models):
            model_df = df_plot[df_plot['model'] == model].copy()
            if model_df.empty:
                continue
            
            model_df['durations'] = model_df['parsed_ts'].apply(get_durations)
            # Pad durations
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
                
            # Add symbol on top
            if model_to_marker and model in model_to_marker:
                marker = model_to_marker[model]
                m_color = 'black'
                if model_to_color and model in model_to_color:
                    m_color = model_to_color[model]
                for j in range(len(unique_tasks)):
                    if bottom[j] > 0:
                        ax.scatter(x_pos[j], bottom[j] + 5, marker=marker, color='black', edgecolor='black', s=30, zorder=5)

        ax.set_ylabel("Average Timesteps")
        ax.set_xticks(x_base)
        ax.set_xticklabels(unique_tasks, rotation=45, ha='right')
        
        # Stage Legend at the bottom
        from matplotlib.patches import Patch
        stage_legend_elements = [Patch(facecolor=stage_colors[i], label=f'Stage {i+1}') for i in range(max_stages)]
        ax.legend(handles=stage_legend_elements, title='Stage Step', loc='upper center',
                  bbox_to_anchor=(0.5, -1.0), ncol=min(max_stages, 5))
        
        buf = io.BytesIO()
        # Manual adjustment to prevent shrinking while making room for labels and legend
        fig.subplots_adjust(bottom=0.35)
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        st.image(buf)
        
    except Exception as e:
        st.error(f"Error plotting average timesteps: {e}")

def plot_bayesian_violin(ax, labels, successes, failures, colors, symbols=None, model_names=None, model_colors=None, title=None, fontsize=20):
    x_positions = np.arange(len(labels))
    successes_arr = np.array(successes, dtype=float).flatten()
    failures_arr = np.array(failures, dtype=float).flatten()
    colors_arr = np.array(colors).flatten()
    symbols_arr = np.array(symbols).flatten() if symbols is not None else None

    # Handle multiple models grouped under the same task label
    # Labels expected to be "TaskName" or "TaskName - ModelName"
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

        # Bayesian update with uniform prior Beta(1,1)
        alpha = S + 1
        beta_param = F + 1

        y = np.linspace(0.001, 0.999, 300)
        density = beta.pdf(y, alpha, beta_param)

        if density.max() == 0:
            continue

        mask = density >= (density.max() * 0.0001)
        y_clipped = y[mask]
        density_clipped = density[mask]

        # Normalize density for plotting width
        x = density_clipped / density.max() * 0.325

        ax.fill_betweenx(y_clipped, x_positions[data_idx] - x, x_positions[data_idx] + x, color=color, alpha=1)

        # Plot the posterior mean
        posterior_mean = alpha / (alpha + beta_param)
        ax.hlines(y=posterior_mean, xmin=x_positions[data_idx] - 0.105, xmax=x_positions[data_idx] + 0.105, color='black', linewidth=2)

        # Plot symbol on top
        if symbols_arr is not None and data_idx < len(symbols_arr):
            marker = symbols_arr[data_idx]
            if marker:
                # Put symbol slightly above the violin
                y_pos = y_clipped.max() + 0.05 if len(y_clipped) > 0 else 1.05
                ax.scatter(x_positions[data_idx], y_pos, marker=marker, color='black', edgecolor='black', s=50, zorder=5)

    # Label groups and draw dividers
    for i, task_label in enumerate(unique_tasks):
        start_idx = unique_indices[i]
        end_idx = unique_indices[i+1] if i + 1 < len(unique_indices) else len(labels)
        group_size = end_idx - start_idx

        if i > 0:
             ax.axvline(x_positions[start_idx] - 0.5, color="gray", linewidth=0.8, linestyle='--')

        center_x = start_idx + (group_size - 1) / 2
        
        # Consistent 45-degree rotation for all labels
        rot = 45
        ha = 'right'
        
        ax.text(center_x, -0.05, task_label, ha=ha, va="top", fontsize=fontsize-2, rotation=rot)

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, 1.15) # Increased to make room for symbols
    ax.set_ylabel("Success Rate", fontsize=fontsize, labelpad=15)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=fontsize+2, y=1.05, fontstyle='italic')
    else:
        ax.set_title("Binary Success Rate (Bayesian Posterior)", fontsize=fontsize+2, y=1.05, fontstyle='italic')

    # Manual adjustment to prevent shrinking while making room for rotated labels
    fig.subplots_adjust(bottom=0.35)
    return ax

def plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols=None, model_names=None, model_colors=None, title=None, ylabel=None, fontsize=12):
    x_positions = np.arange(len(labels))
    
    # Extract group names for labeling and dividers
    group_names = [l.split(' - ')[0] for l in labels]
    unique_groups = []
    unique_indices = []
    for i, g in enumerate(group_names):
        if g not in unique_groups:
            unique_groups.append(g)
            unique_indices.append(i)
            
    # Plot bars
    ax.bar(x_positions, values, color=colors, edgecolor='none')
    
    # Plot symbols on top
    if symbols:
        for i, (v, s) in enumerate(zip(values, symbols)):
            if s:
                ax.scatter(x_positions[i], v + 0.05, marker=s, color='black', edgecolor='black', s=50, zorder=5)
                
    # Dividers and labels
    for i, group_label in enumerate(unique_groups):
        start_idx = unique_indices[i]
        end_idx = unique_indices[i+1] if i + 1 < len(unique_indices) else len(labels)
        group_size = end_idx - start_idx
        
        if i > 0:
            ax.axvline(x_positions[start_idx] - 0.5, color="gray", linewidth=0.8, linestyle='--')
            
        center_x = start_idx + (group_size - 1) / 2
        # Consistent 45-degree rotation for all labels
        rot = 45
        ha = 'right'
        ax.text(center_x, -0.05, group_label, ha=ha, va="top", fontsize=fontsize-2, rotation=rot)
        
    ax.set_xlim(-0.5, len(labels) - 0.5)
    # Automatically adjust ylim to fit symbols
    max_val = max(values) if values else 1.0
    ax.set_ylim(0, max(1.1, max_val + 0.15))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=fontsize+2)
        
    # Manual adjustment to prevent shrinking while making room for rotated labels
    fig.subplots_adjust(bottom=0.35)
    return ax

def render_model_legend(model_to_marker, model_to_color):
    """Render a standalone legend infographic for models."""
    if not model_to_marker:
        return
    
    unique_models = sorted(model_to_marker.keys())
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    # Create a wide, short figure
    fig, ax = plt.subplots(figsize=(10, 0.8))
    ax.axis('off')
    
    # Use two handles for each model: a patch for the bar/violin color and a Line2D for the marker shape
    legend_elements = []
    for m in unique_models:
        color = model_to_color[m]
        marker = model_to_marker[m]
        # Combination of color patch and marker shape
        legend_elements.append((Patch(facecolor=color, edgecolor='black', alpha=0.8),
                               Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', 
                                      markeredgecolor='black', markersize=10)))
    
    # Handler for the tuple of handles
    from matplotlib.legend_handler import HandlerTuple
    ax.legend(handles=legend_elements, labels=unique_models, loc='center', ncol=min(len(unique_models), 5), 
              title="Models", frameon=False, fontsize=10, handler_map={tuple: HandlerTuple(ndivide=None)})
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    plt.close(fig)
    st.image(buf)

def render_metric_bar_chart(df, metric_col, title, ylabel, model_to_marker, model_to_color, color=None):
    """Helper to render a grouped bar chart for a specific metric."""
    if metric_col not in df.columns:
        st.info(f"Column '{metric_col}' not found in data.")
        return

    try:
        # Filter for rows where the metric is not NaN
        metric_df = df[df[metric_col].notna()].copy()
        if metric_df.empty:
            st.info(f"No data available for metric '{metric_col}'.")
            return

        # Group by task and model
        stats = metric_df.groupby(['task', 'model'])[metric_col].mean().reset_index()
        stats = stats.sort_values(['task', 'model'])
        
        labels = [f"{row['task']} - {row['model']}" for _, row in stats.iterrows()]
        values = stats[metric_col].tolist()
        
        # Use model specific colors if provided, else fallback to passed color or default gold
        if model_to_color:
            colors = [model_to_color[row['model']] for _, row in stats.iterrows()]
        else:
            colors = [color if color else 'gold'] * len(labels)
            
        symbols = [model_to_marker[row['model']] for _, row in stats.iterrows()]
        model_names = stats['model'].tolist()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel=ylabel, title=title)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        st.image(buf)
    except Exception as e:
        st.error(f"Error plotting {title}: {e}")

# ===== SIDEBAR =====
# Sidebar Filters (Moved to top)
st.sidebar.header("Filter Data & Videos")

with st.sidebar.expander("Filter by Task", expanded=False):
    selected_tasks = []
    all_tasks = st.checkbox("All Tasks", value=False)
    if all_tasks:
        selected_tasks = dashboard_utils.SUPPORTED_TASKS
    else:
        for task in dashboard_utils.SUPPORTED_TASKS:
            if st.checkbox(task, key=f"chk_task_{task}"):
                selected_tasks.append(task)

with st.sidebar.expander("Filter by Perturbation", expanded=False):
    selected_perts = []
    all_perts = st.checkbox("All Perturbations", value=False)
    if all_perts:
        selected_perts = dashboard_utils.SUPPORTED_PERTURBATIONS
    else:
        for pert in dashboard_utils.SUPPORTED_PERTURBATIONS:
            if st.checkbox(pert, key=f"chk_pert_{pert}"):
                selected_perts.append(pert)

st.sidebar.markdown("---")

st.sidebar.title("Experiment Browser")
search_query = st.sidebar.text_input("Search Experiments", key="search_query", placeholder="e.g. baseline")
render_tree(LOGS_DIR, search_query=search_query)

# Collect selected runs from checkboxes
selected_runs = [key[4:] for key, value in st.session_state.items() if key.startswith("chk_") and value and os.path.isdir(key[4:]) and get_cached_is_experiment_folder(key[4:])]

# ===== MAIN CONTENT =====
if selected_runs:
    # Reset video page when experiment selection changes
    if sorted(selected_runs) != st.session_state.get('last_selected_runs', []):
        st.session_state.video_page = 0
        st.session_state.last_selected_runs = sorted(selected_runs)

    # --- Header ---
    st.title("Experiment Dashboard")
    
    # Get common path prefix to identify experiment and model
    if len(selected_runs) == 1:
        rel_path = os.path.relpath(selected_runs[0], LOGS_DIR)
        path_parts = rel_path.split(os.sep)
        experiment_name = path_parts[0] if len(path_parts) > 0 else "N/A"
        model_name = path_parts[1] if len(path_parts) > 1 else "N/A"
    else:
        # Try to find common parts
        common_path = os.path.commonpath(selected_runs)
        rel_common_path = os.path.relpath(common_path, LOGS_DIR)
        path_parts = rel_common_path.split(os.sep)
        experiment_name = path_parts[0] if len(path_parts) > 0 and path_parts[0] != '.' else "Multiple Experiments"
        model_name = path_parts[1] if len(path_parts) > 1 else "Multiple Models"
    
    run_ids = [os.path.basename(run) for run in selected_runs]
    
    st.markdown(f"<h3><b>Experiment:</b> {experiment_name}</h3><h3><b>Model:</b> {model_name}</h3>", unsafe_allow_html=True)
    if len(run_ids) == 1:
        st.markdown(f"<h3><b>Run ID:</b> {run_ids[0]}</h3>", unsafe_allow_html=True)
    else:
        with st.expander(f"Selected {len(run_ids)} Runs"):
            st.write(", ".join(run_ids))

    st.divider()

    # --- Data Loading and Aggregation ---
    all_reports = []
    all_videos = []
    for run_path in selected_runs:
        report = get_cached_reports(run_path)
        if report is not None:
            # Add model info
            rel_path = os.path.relpath(run_path, LOGS_DIR)
            path_parts = rel_path.split(os.sep)
            model_name = path_parts[1] if len(path_parts) > 1 else "Unknown"
            report['model'] = model_name
            all_reports.append(report)
        all_videos.extend(get_cached_videos(run_path))
    
    if not all_reports:
        st.warning("No reports found for the selected run(s).")
        st.stop()
        
    df = pd.concat(all_reports, ignore_index=True)

    # Apply filters to dataframe
    df = dashboard_utils.filter_dataframe(df, selected_tasks, selected_perts)

    # --- Experiment Status ---
    st.header("Experiment Status")
    try:
        # For status, we check the first selected run's metadata
        first_run_path = selected_runs[0]
        exp_path_for_meta = os.path.dirname(os.path.dirname(first_run_path))
        
        metadata, err = get_cached_experiment_metadata(exp_path_for_meta)

        if metadata:
            tasks_indices = metadata.get("task_ids", [])
            perts_indices = metadata.get("perturbation_ids", [])
            required_repeats = metadata.get("repeats", 0) * len(selected_runs) # Scale repeats by number of runs

            st.markdown(f"**Target Configuration (from {os.path.basename(exp_path_for_meta)}/metadata.json):**\n"
                        f"* Tasks: {tasks_indices}\n"
                        f"* Perturbations: {perts_indices}\n"
                        f"* Repeats per run: {metadata.get('repeats', 0)}")

            status, msg = dashboard_utils.check_experiment_status(df, tasks_indices, perts_indices, required_repeats)

            if status:
                st.success("✅ All required tasks and perturbations evaluated with sufficient samples across all selected runs.")
            else:
                st.error("❌ Missing evaluations.")
                with st.expander("Missing Combinations", expanded=False):
                    missing_lines = msg.split("Missing evaluations:\n")
                    if len(missing_lines) > 1:
                        missing_items = missing_lines[1].split("\n")
                        formatted_missing = "\n".join([f"- {item}" for item in missing_items if item.strip()])
                        st.write(formatted_missing)
                    else:
                         st.write(msg)
        else:
            st.warning(f"Could not load metadata to check experiment status. {err}")

    except Exception as e:
        st.error(f"Error in Experiment Status: {e}")

    st.divider()

    # --- Model Legend (Global) ---
    if df is not None and not df.empty:
        unique_models = sorted(df['model'].unique())
        model_to_marker = {model: MARKERS[i % len(MARKERS)] for i, model in enumerate(unique_models)}
        model_to_color = {model: COLORS[i % len(COLORS)] for i, model in enumerate(unique_models)}
        render_model_legend(model_to_marker, model_to_color)

    # --- Plots ---
    st.header("Success Metrics")
    try:
        if df is not None and not df.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Success Rate per Task")
                if 'binary_SR' in df.columns:
                    # Group by both task and model
                    task_stats = df.groupby(['task', 'model'])['binary_SR'].agg(['sum', 'count']).reset_index()
                    task_stats = task_stats.sort_values(['task', 'model'])
                    
                    labels = [f"{row['task']} - {row['model']}" for _, row in task_stats.iterrows()]
                    successes = task_stats['sum'].tolist()
                    failures = (task_stats['count'] - task_stats['sum']).tolist()
                    colors = [model_to_color[row['model']] for _, row in task_stats.iterrows()]
                    symbols = [model_to_marker[row['model']] for _, row in task_stats.iterrows()]
                    model_names = task_stats['model'].tolist()

                    # Add mean over all tasks per model
                    all_tasks_stats = df.groupby('model')['binary_SR'].agg(['sum', 'count']).reset_index()
                    all_tasks_stats = all_tasks_stats.sort_values('model')
                    for _, row in all_tasks_stats.iterrows():
                        labels.append(f"All Tasks (Mean) - {row['model']}")
                        successes.append(row['sum'])
                        failures.append(row['count'] - row['sum'])
                        colors.append(model_to_color[row['model']])
                        symbols.append(model_to_marker[row['model']])
                        model_names.append(row['model'])

                    fig, ax = plt.subplots(figsize=(6, 4))
                    plot_bayesian_violin(ax, labels, successes, failures, colors, symbols=symbols, model_names=model_names, model_colors=model_to_color, fontsize=12)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    plt.close(fig)
                    st.image(buf)

            with c2:
                st.subheader("Success Rate per Perturbation")
                df['clean_pert'] = df['perturbation'].apply(lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x))

                if 'binary_SR' in df.columns:
                    pert_stats = df.groupby(['clean_pert', 'model'])['binary_SR'].agg(['sum', 'count']).reset_index()
                    pert_stats = pert_stats.sort_values(['clean_pert', 'model'])
                    
                    labels = [f"{row['clean_pert']} - {row['model']}" for _, row in pert_stats.iterrows()]
                    successes = pert_stats['sum'].tolist()
                    failures = (pert_stats['count'] - pert_stats['sum']).tolist()
                    colors = [model_to_color[row['model']] for _, row in pert_stats.iterrows()]
                    symbols = [model_to_marker[row['model']] for _, row in pert_stats.iterrows()]
                    model_names = pert_stats['model'].tolist()

                    fig, ax = plt.subplots(figsize=(6, 4))
                    plot_bayesian_violin(ax, labels, successes, failures, colors, symbols=symbols, model_names=model_names, model_colors=model_to_color, fontsize=12)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    plt.close(fig)
                    st.image(buf)
                else:
                    st.info("No binary_SR column found for plots.")
            
            st.divider()
            c3, c4 = st.columns(2)

            with c3:
                st.subheader("Task Progression per Task")
                if 'task_progression' in df.columns:
                    task_prog = df.groupby(['task', 'model'])['task_progression'].mean().reset_index()
                    task_prog = task_prog.sort_values(['task', 'model'])
                    
                    labels = [f"{row['task']} - {row['model']}" for _, row in task_prog.iterrows()]
                    values = task_prog['task_progression'].tolist()
                    colors = [model_to_color[row['model']] for _, row in task_prog.iterrows()]
                    symbols = [model_to_marker[row['model']] for _, row in task_prog.iterrows()]
                    model_names = task_prog['model'].tolist()

                    # Add mean over all tasks per model
                    all_tasks_prog = df.groupby('model')['task_progression'].mean().reset_index()
                    all_tasks_prog = all_tasks_prog.sort_values('model')
                    for _, row in all_tasks_prog.iterrows():
                        labels.append(f"All Tasks (Mean) - {row['model']}")
                        values.append(row['task_progression'])
                        colors.append(model_to_color[row['model']])
                        symbols.append(model_to_marker[row['model']])
                        model_names.append(row['model'])

                    fig, ax = plt.subplots(figsize=(6, 4))
                    plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel="Task Progression")

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    plt.close(fig)
                    st.image(buf)
                else:
                    st.info("No task_progression column found for plots.")
            
            with c4:
                st.subheader("Task Progression per Perturbation")
                if 'task_progression' in df.columns:
                    pert_prog = df.groupby(['clean_pert', 'model'])['task_progression'].mean().reset_index()
                    pert_prog = pert_prog.sort_values(['clean_pert', 'model'])
                    
                    labels = [f"{row['clean_pert']} - {row['model']}" for _, row in pert_prog.iterrows()]
                    values = pert_prog['task_progression'].tolist()
                    colors = [model_to_color[row['model']] for _, row in pert_prog.iterrows()]
                    symbols = [model_to_marker[row['model']] for _, row in pert_prog.iterrows()]
                    model_names = pert_prog['model'].tolist()

                    fig, ax = plt.subplots(figsize=(6, 4))
                    plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel="Task Progression")

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    plt.close(fig)
                    st.image(buf)
                else:
                    st.info("No task_progression column found for plots.")

            st.divider()

            c5, c6 = st.columns(2)
            with c5:
                st.subheader("Failure Stage Frequency")
                plot_stage_frequency(df, model_to_marker=model_to_marker, model_to_color=model_to_color)
            
            with c6:
                st.subheader("Failure Stage Frequency per Task")
                plot_stage_frequency_per_task(df, model_to_marker=model_to_marker, model_to_color=model_to_color)
            
            st.caption("*Note: Data for perturbation 'SB-VRB' is excluded from the 'Failure Stage Frequency per Task' plot.*")
                
            st.divider()

            # --- Auxiliary Metrics ---
            st.header("Auxiliary metrics")
            
            c7, c8 = st.columns(2)
            with c7:
                st.subheader("Time to Completion per Task (Successful Only)")
                if 'task_progression' in df.columns:
                    # Look for the correct timestamp column
                    col_name = None
                    for col in ['task_progression_timesteps', 'task_progression_timestamps', 'task_progression_timstamps']:
                        if col in df.columns:
                            col_name = col
                            break
                    if col_name:
                        # Filter for rows where task_progression is 1
                        success_df = df[df['task_progression'] == 1].copy()
                        if not success_df.empty:
                            try:
                                # Parse timestamps
                                success_df['parsed_ts'] = success_df[col_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                                # Extract the last timestamp
                                success_df['completion_time'] = success_df['parsed_ts'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else 0)
                                
                                task_time = success_df.groupby(['task', 'model'])['completion_time'].mean().reset_index()
                                task_time = task_time.sort_values(['task', 'model'])
                                
                                labels = [f"{row['task']} - {row['model']}" for _, row in task_time.iterrows()]
                                values = task_time['completion_time'].tolist()
                                colors = [model_to_color[row['model']] for _, row in task_time.iterrows()]
                                symbols = [model_to_marker[row['model']] for _, row in task_time.iterrows()]
                                model_names = task_time['model'].tolist()
                                
                                fig, ax = plt.subplots(figsize=(6, 4))
                                plot_grouped_bars_with_symbols(ax, labels, values, colors, symbols, model_names, model_colors=model_to_color, ylabel="Average Completion Time")
                                
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", bbox_inches='tight')
                                plt.close(fig)
                                st.image(buf)
                            except Exception as e:
                                st.error(f"Error parsing timestamps: {e}")
                        else:
                            st.info("No successful runs (task_progression == 1) to calculate time to completion.")
                    else:
                        st.info("Required column for timestamps not found.")
                else:
                    st.info("Required column 'task_progression' not found.")
            
            with c8:
                st.subheader("Stage Timesteps per Task")
                plot_task_progression_timesteps_per_task(df, model_to_marker=model_to_marker, model_to_color=model_to_color)
            
            st.caption("*Note: Data for perturbation 'SB-VRB' is excluded from the 'Stage Timesteps per Task' plot.*")
            
            st.divider()

            # --- Physical and Kinematic Metrics ---
            c9, c10 = st.columns(2)
            with c9:
                render_metric_bar_chart(df, 'joint_vel_var', "Joint Velocity Variance", "Mean Variance", model_to_marker, model_to_color, color='lightblue')
            with c10:
                render_metric_bar_chart(df, 'joint_acc_var', "Joint Acceleration Variance", "Mean Variance", model_to_marker, model_to_color, color='lightblue')

            st.divider()

            c11, c12 = st.columns(2)
            with c11:
                render_metric_bar_chart(df, 'joint_jerk', "Joint Jerk", "Mean Jerk", model_to_marker, model_to_color, color='lightblue')
            with c12:
                render_metric_bar_chart(df, 'cart_jerk', "Cartesian Jerk", "Mean Jerk", model_to_marker, model_to_color, color='lightblue')

            st.divider()

            c13, c14 = st.columns(2)
            with c13:
                render_metric_bar_chart(df, 'joint_path_length', "Joint Path Length", "Mean Path Length", model_to_marker, model_to_color, color='lightgreen')
            with c14:
                render_metric_bar_chart(df, 'cart_path_length', "Cartesian Path Length", "Mean Path Length", model_to_marker, model_to_color, color='lightgreen')

            st.divider()

            # --- Interaction and Collision Metrics ---
            c15, c16 = st.columns(2)
            with c15:
                render_metric_bar_chart(df, 'collisions_env', "Environment Collisions", "Count", model_to_marker, model_to_color, color='salmon')
            with c16:
                render_metric_bar_chart(df, 'object_drops', "Object Drops", "Count", model_to_marker, model_to_color, color='salmon')

        else:
            if df is not None:
                st.info("No data matches the selected filters.")
            else:
                st.info("No data available.")
    except Exception as e:
        st.error(f"Error in Plots: {e}")

    # --- Aggregated Reports ---
    st.header("Aggregated Reports")
    try:
        if df is not None:
            st.write(f"Showing {len(df)} rows.")
            st.dataframe(df, height=300)
        else:
            st.info("No reports found.")
    except Exception as e:
        st.error(f"Error in Aggregated Reports: {e}")

    # --- Videos (paginated fragment) ---
    st.header("Videos")
    render_video_section(all_videos, selected_tasks, selected_perts)
else:
    if not os.path.exists(LOGS_DIR):
        st.error(f"Logs directory '{LOGS_DIR}' not found.")
    else:
        st.info("Please select an experiment from the sidebar.")
