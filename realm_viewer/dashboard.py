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

def plot_stage_frequency(df):
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
    
    # Get frequency (counts) of each simplified stage
    stage_counts = df_plot['simplified_stage'].value_counts()
    
    total_rows = len(df_plot)
    if total_rows == 0:
        st.info("No data to plot stage frequency.")
        return

    # Calculate proportions
    proportions = [stage_counts.get(l, 0) / total_rows for l in ordered_labels]
    
    colors = [color_dict[label] for label in ordered_labels]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(ordered_labels, proportions, color=colors, edgecolor='none')
    ax.set_ylim(0, 1) # Set y-axis range from 0 to 1 as requested
    ax.set_ylabel("Frequency (Proportion of Total Data)")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha='right')
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    st.image(buf)

def plot_stage_frequency_per_task(df):
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

    # Create a crosstab of task vs simplified_stage and normalize by row to get frequencies
    ct = pd.crosstab(df_plot['task'], df_plot['simplified_stage'], normalize='index')
    
    # Ensure all ordered stages are present in ct to maintain consistency
    for stage in ordered_stages:
        if stage not in ct.columns:
            ct[stage] = 0.0
            
    # Reorder the columns of ct according to the globally calculated task progression order
    ct = ct[ordered_stages]

    plot_colors = [color_dict[col] for col in ordered_stages]

    # Use the color map to create legend elements in the same order
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_dict[stage], label=stage) for stage in ordered_stages]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot stacked bar chart using the ordered columns and mapped colors
    # Added linewidth=0 to prevent drawing visible edges for 0-height segments
    ct.plot(kind='bar', stacked=True, ax=ax, color=plot_colors, linewidth=0)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha='right')
    
    # Move legend outside, using custom legend elements to enforce matching color/label
    ax.legend(handles=legend_elements, title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    st.image(buf)
    st.caption("*Note: Data for perturbation 'SB-VRB' is excluded from this plot.*")

def plot_task_progression_timesteps_per_task(df):
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
    
    if df_plot.empty:
        st.info("No data available to plot average timesteps.")
        return

    try:
        # Parse timestamps into lists
        df_plot['parsed_ts'] = df_plot[col_name].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Determine the maximum number of stages across all tasks
        max_stages = df_plot['parsed_ts'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
        
        if max_stages == 0:
            st.info("No valid timestamp data found.")
            return
            
        # Calculate the duration of each stage. 
        # ts[0] is time to reach stage 1, ts[1]-ts[0] is time spent in stage 1 to reach stage 2, etc.
        def get_durations(ts_list):
            if not isinstance(ts_list, list) or len(ts_list) == 0:
                return []
            durations = []
            prev_ts = 0
            for ts in ts_list:
                durations.append(ts - prev_ts)
                prev_ts = ts
            return durations

        df_plot['durations'] = df_plot['parsed_ts'].apply(get_durations)
        
        # Pad with NaNs so all lists have length max_stages
        df_plot['durations'] = df_plot['durations'].apply(lambda x: x + [np.nan] * (max_stages - len(x)))

        # Expand durations into separate columns
        durations_df = pd.DataFrame(df_plot['durations'].tolist(), index=df_plot.index)
        
        # Rename columns to 'Stage 1', 'Stage 2', etc.
        durations_df.columns = [f'Stage {i+1}' for i in range(max_stages)]
        
        # Add task column back
        durations_df['task'] = df_plot['task']
        
        # Calculate mean duration per stage per task
        mean_durations = durations_df.groupby('task').mean()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot stacked bar chart
        # We can use a colormap for the stages
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / max_stages) for i in range(max_stages)]
        
        # Added linewidth=0 to prevent drawing visible edges for 0-height segments
        mean_durations.plot(kind='bar', stacked=True, ax=ax, color=colors, linewidth=0)
        
        ax.set_ylabel("Average Timesteps")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha='right')
        
        # Move legend outside
        ax.legend(title='Stage Step', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        st.image(buf)
        st.caption("*Note: Data for perturbation 'SB-VRB' is excluded from this plot.*")
        
    except Exception as e:
        st.error(f"Error plotting average timesteps: {e}")

def plot_bayesian_violin(ax, labels, successes, failures, colors, title=None, fontsize=20):
    x_positions = np.arange(len(labels))
    successes_arr = np.array(successes, dtype=float).flatten()
    failures_arr = np.array(failures, dtype=float).flatten()
    colors_arr = np.array(colors).flatten()

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

    # Label groups and draw dividers
    for i, task_label in enumerate(unique_tasks):
        start_idx = unique_indices[i]
        end_idx = unique_indices[i+1] if i + 1 < len(unique_indices) else len(labels)
        group_size = end_idx - start_idx

        if i > 0:
             ax.axvline(x_positions[start_idx] - 0.5, color="gray", linewidth=0.8, linestyle='--')

        center_x = start_idx + (group_size - 1) / 2
        
        # Determine rotation based on label length and number of tasks
        rot = 45 if (len(task_label) > 10 or len(labels) > 5) else 0
        ha = 'right' if rot != 0 else 'center'
        
        ax.text(center_x, -0.05, task_label, ha=ha, va="top", fontsize=fontsize-2, rotation=rot)

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success Rate", fontsize=fontsize, labelpad=15)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=fontsize+2, y=1.05, fontstyle='italic')
    else:
        ax.set_title("Binary Success Rate (Bayesian Posterior)", fontsize=fontsize+2, y=1.05, fontstyle='italic')

    return ax

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

    # --- Plots ---
    st.header("Success Metrics")
    try:
        if df is not None and not df.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Success Rate per Task")
                if 'binary_SR' in df.columns:
                    task_stats = df.groupby('task')['binary_SR'].agg(['sum', 'count']).reset_index()
                    labels = task_stats['task'].tolist()
                    successes = task_stats['sum'].tolist()
                    failures = (task_stats['count'] - task_stats['sum']).tolist()
                    colors = ['lightgreen'] * len(labels)

                    # Add mean over all tasks
                    all_successes = df['binary_SR'].sum()
                    all_count = df['binary_SR'].count()
                    labels.append("All Tasks (Mean)")
                    successes.append(all_successes)
                    failures.append(all_count - all_successes)
                    colors.append('darkgreen')

                    fig, ax = plt.subplots(figsize=(6, 5))
                    plot_bayesian_violin(ax, labels, successes, failures, colors, fontsize=12)

                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    st.image(buf)

            with c2:
                st.subheader("Success Rate per Perturbation")
                df['clean_pert'] = df['perturbation'].apply(lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x))

                if 'binary_SR' in df.columns:
                    pert_stats = df.groupby('clean_pert')['binary_SR'].agg(['sum', 'count']).reset_index()
                    labels = pert_stats['clean_pert'].tolist()
                    successes = pert_stats['sum'].tolist()
                    failures = (pert_stats['count'] - pert_stats['sum']).tolist()
                    colors = ['lightgreen'] * len(labels)

                    fig, ax = plt.subplots(figsize=(6, 5))
                    plot_bayesian_violin(ax, labels, successes, failures, colors, fontsize=12)

                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    st.image(buf)
                else:
                    st.info("No binary_SR column found for plots.")
            
            st.divider()
            c3, c4 = st.columns(2)

            with c3:
                st.subheader("Task Progression per Task")
                if 'task_progression' in df.columns:
                    task_prog = df.groupby('task')['task_progression'].mean().reset_index()
                    
                    # Add mean over all tasks
                    mean_val = df['task_progression'].mean()
                    task_prog = pd.concat([task_prog, pd.DataFrame([{'task': 'All Tasks (Mean)', 'task_progression': mean_val}])], ignore_index=True)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    
                    colors = ['lightblue' if task != 'All Tasks (Mean)' else 'steelblue' for task in task_prog['task']]
                    ax.bar(task_prog['task'], task_prog['task_progression'], color=colors)
                    
                    # Add dashed horizontal line for mean
                    ax.axhline(y=mean_val, color='steelblue', linestyle='--', linewidth=1)
                    
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Task Progression")
                    ax.set_xlabel("")
                    plt.xticks(rotation=45, ha='right')
                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    st.image(buf)
                else:
                    st.info("No task_progression column found for plots.")
            
            with c4:
                st.subheader("Task Progression per Perturbation")
                if 'task_progression' in df.columns:
                    pert_prog = df.groupby('clean_pert')['task_progression'].mean().reset_index()
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(pert_prog['clean_pert'], pert_prog['task_progression'], color='lightblue')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Task Progression")
                    ax.set_xlabel("")
                    plt.xticks(rotation=45, ha='right')
                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    st.image(buf)
                else:
                    st.info("No task_progression column found for plots.")

            st.divider()

            c5, c6 = st.columns(2)
            with c5:
                st.subheader("Failure Stage Frequency")
                plot_stage_frequency(df)
            
            with c6:
                st.subheader("Failure Stage Frequency per Task")
                plot_stage_frequency_per_task(df)
                
            st.divider()
            
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
                                
                                task_time = success_df.groupby('task')['completion_time'].mean().reset_index()
                                fig, ax = plt.subplots(figsize=(5, 4))
                                ax.bar(task_time['task'], task_time['completion_time'], color='gold')
                                ax.set_ylabel("Average Completion Time")
                                ax.set_xlabel("")
                                plt.xticks(rotation=45, ha='right')
                                buf = io.BytesIO()
                                fig.tight_layout()
                                fig.savefig(buf, format="png")
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
                plot_task_progression_timesteps_per_task(df)

        else:
            if raw_df is not None:
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
