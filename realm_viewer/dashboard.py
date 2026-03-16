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
if "selected_experiment" not in st.session_state:
    st.session_state.selected_experiment = None
if "expanded_folders" not in st.session_state:
    st.session_state.expanded_folders = set()
if "video_page" not in st.session_state:
    st.session_state.video_page = 0
if "last_experiment_viewed" not in st.session_state:
    st.session_state.last_experiment_viewed = None

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

def _select_experiment(path, has_children):
    st.session_state.selected_experiment = path
    if has_children:
        _toggle_folder(path)

def has_experiments_recursively(base_path):
    """Check if the given path or any of its subdirectories contain an experiment."""
    if get_cached_is_experiment_folder(base_path):
        return True
    
    subdirs = get_cached_subdirectories(base_path)
    for d in subdirs:
        full_path = os.path.join(base_path, d)
        if has_experiments_recursively(full_path):
            return True
    return False

# --- Tree rendering ---
def render_tree(base_path, depth=0):
    """Render a collapsible directory tree in the sidebar with proper indentation."""
    if depth > 5:
        return

    subdirs = get_cached_subdirectories(base_path)
    if not subdirs:
        return

    for d in subdirs:
        full_path = os.path.join(base_path, d)
        
        # Check if this branch has any experiments
        if not has_experiments_recursively(full_path):
            continue
            
        is_exp = get_cached_is_experiment_folder(full_path)
        sub_subdirs = get_cached_subdirectories(full_path)
        
        # Do not treat experiment folders as having children in the tree to prevent them from rolling down
        has_children = len(sub_subdirs) > 0 and not is_exp
        
        is_expanded = full_path in st.session_state.expanded_folders
        is_selected = st.session_state.selected_experiment == full_path

        # Build indented label: 2 em-spaces per depth level
        indent = "\u2003" * (depth * 2)

        if has_children:
            chevron = "▾" if is_expanded else "▸"
        else:
            chevron = "\u2003"

        if is_exp:
            marker = "● " if is_selected else ""
            label = f"{indent}{chevron} {marker}🔬 {d}"
            st.sidebar.button(
                label,
                key=f"tree_{full_path}",
                on_click=_select_experiment,
                args=(full_path, False),
            )
        elif has_children:
            label = f"{indent}{chevron} 📁 {d}"
            st.sidebar.button(
                label,
                key=f"tree_{full_path}",
                on_click=_toggle_folder,
                args=(full_path,),
            )
        else:
            # Non-experiment leaf folder (should not be reached if has_experiments_recursively is working correctly)
            st.sidebar.markdown(
                f"<div style='padding-left:{depth * 1.5}rem; font-size:0.85rem; color:gray; "
                f"padding-top:0.1rem; padding-bottom:0.1rem;'>\u2003 📁 {d}</div>",
                unsafe_allow_html=True,
            )

        # Recurse into expanded children
        if has_children and is_expanded:
            render_tree(full_path, depth + 1)

# --- Video page callbacks ---
def _video_prev_page():
    st.session_state.video_page = max(0, st.session_state.video_page - 1)

def _video_next_page():
    st.session_state.video_page += 1

# --- Video section (fragment for partial reruns) ---
@st.fragment
def render_video_section(selected_path, selected_tasks, selected_perts):
    """Render paginated video gallery. Runs as a fragment so page navigation
    does not trigger a full dashboard rerun."""
    videos = get_cached_videos(selected_path)
    filtered_videos = dashboard_utils.filter_videos(videos, selected_tasks, selected_perts)

    if not filtered_videos:
        if videos:
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
            st.caption(os.path.basename(video_path))

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
    ax.bar(ordered_labels, proportions, color=colors)
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

    df_plot = df.copy()
    
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
    ct.plot(kind='bar', stacked=True, ax=ax, color=plot_colors)
    
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

    # Use all rows, not just successful ones
    df_plot = df.copy()
    
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
        
        mean_durations.plot(kind='bar', stacked=True, ax=ax, color=colors)
        
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
        
    except Exception as e:
        st.error(f"Error plotting average timesteps: {e}")

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
render_tree(LOGS_DIR)

# ===== MAIN CONTENT =====
if st.session_state.selected_experiment and os.path.exists(st.session_state.selected_experiment):
    # Reset video page when experiment changes
    if st.session_state.selected_experiment != st.session_state.last_experiment_viewed:
        st.session_state.video_page = 0
        st.session_state.last_experiment_viewed = st.session_state.selected_experiment

    selected_path = st.session_state.selected_experiment

    # Header Parsing
    rel_path = os.path.relpath(selected_path, LOGS_DIR)
    path_parts = rel_path.split(os.sep)

    experiment_name = path_parts[0] if len(path_parts) > 0 else "N/A"
    model_name = path_parts[1] if len(path_parts) > 1 else "N/A"
    run_id = path_parts[2] if len(path_parts) > 2 else "N/A"

    st.title("Experiment Dashboard")

    st.markdown(f"<h3><b>Experiment:</b> {experiment_name}</h3><h3><b>Model:</b> {model_name}</h3><h3><b>Run ID:</b> {run_id}</h3>", unsafe_allow_html=True)

    st.divider()

    # Load Reports
    raw_df = get_cached_reports(selected_path)

    # Apply filters to dataframe
    df = dashboard_utils.filter_dataframe(raw_df, selected_tasks, selected_perts)

    # --- Experiment Status ---
    st.header("Experiment Status")
    try:
        rel_path = os.path.relpath(selected_path, LOGS_DIR)
        parts = rel_path.split(os.sep)

        if len(parts) > 0:
            experiment_name = parts[0]
            experiment_path = os.path.join(LOGS_DIR, experiment_name)

            metadata, err = get_cached_experiment_metadata(experiment_path)

            if metadata:
                tasks_indices = metadata.get("task_ids", [])
                perts_indices = metadata.get("perturbation_ids", [])
                required_repeats = metadata.get("repeats", 0)

                st.markdown(f"**Target Configuration (from {experiment_name}/metadata.json):**\n"
                            f"* Tasks: {tasks_indices}\n"
                            f"* Perturbations: {perts_indices}\n"
                            f"* Repeats: {required_repeats}")

                status, msg = dashboard_utils.check_experiment_status(raw_df, tasks_indices, perts_indices, required_repeats)

                if status:
                    st.success("✅ All required tasks and perturbations evaluated with sufficient samples.")
                else:
                    st.error("❌ Missing evaluations.")
                    with st.expander("Missing Combinations", expanded=False):
                        # The original error message returned from check_experiment_status starts with "Missing evaluations:\n"
                        # We split it to only get the actual missing combinations
                        missing_lines = msg.split("Missing evaluations:\n")
                        if len(missing_lines) > 1:
                            # Reformat the output to bullet points
                            missing_items = missing_lines[1].split("\n")
                            formatted_missing = "\n".join([f"- {item}" for item in missing_items if item.strip()])
                            st.write(formatted_missing)
                        else:
                             st.write(msg)

                # Show completed combinations (from RAW data)
                completed = dashboard_utils.get_completed_experiments(raw_df, required_repeats)
                if completed:
                    with st.expander("Completed Combinations", expanded=False):
                        grouped = defaultdict(list)
                        for t, p in completed:
                            grouped[t].append(p)

                        for t, perts in grouped.items():
                            st.write(f"- **{t}**: {', '.join(perts)}")
            else:
                st.warning(err)
        else:
            st.warning("Could not determine experiment directory.")

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
                    task_sr = df.groupby('task')['binary_SR'].mean().reset_index()

                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(task_sr['task'], task_sr['binary_SR'], color='lightgreen')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Success Rate")
                    ax.set_xlabel("")
                    plt.xticks(rotation=45, ha='right')

                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    st.image(buf)

            with c2:
                st.subheader("Success Rate per Perturbation")
                df['clean_pert'] = df['perturbation'].apply(lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x))

                if 'binary_SR' in df.columns:
                    pert_sr = df.groupby('clean_pert')['binary_SR'].mean().reset_index()

                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(pert_sr['clean_pert'], pert_sr['binary_SR'], color='skyblue')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Success Rate")
                    ax.set_xlabel("")
                    plt.xticks(rotation=45, ha='right')

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
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(task_prog['task'], task_prog['task_progression'], color='lightcoral')
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
                    ax.bar(pert_prog['clean_pert'], pert_prog['task_progression'], color='plum')
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
                st.subheader("Stage Frequency per Task")
                plot_stage_frequency_per_task(df)
                
            st.divider()
            
            c7, c8 = st.columns(2)
            with c7:
                st.subheader("Time to Completion per Task")
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
    render_video_section(selected_path, selected_tasks, selected_perts)
else:
    if not os.path.exists(LOGS_DIR):
        st.error(f"Logs directory '{LOGS_DIR}' not found.")
    else:
        st.info("Please select an experiment from the sidebar.")
