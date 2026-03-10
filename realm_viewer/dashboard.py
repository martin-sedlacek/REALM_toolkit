import streamlit as st
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import dashboard_utils
from collections import defaultdict

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
        is_exp = get_cached_is_experiment_folder(full_path)
        sub_subdirs = get_cached_subdirectories(full_path)
        has_children = len(sub_subdirs) > 0
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
                args=(full_path, has_children),
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
            # Non-experiment leaf folder
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

# ===== SIDEBAR =====
st.sidebar.title("Experiment Browser")
render_tree(LOGS_DIR)

# Sidebar Filters
st.sidebar.markdown("---")
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiment", experiment_name)
    c2.metric("Model", model_name)
    c3.metric("Run ID", run_id)

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

                st.write(f"**Target Configuration (from {experiment_name}/metadata.json):** Tasks: {tasks_indices}, Perturbations: {perts_indices}, Repeats: {required_repeats}")

                status, msg = dashboard_utils.check_experiment_status(raw_df, tasks_indices, perts_indices, required_repeats)

                if status:
                    st.success("✅ " + msg)
                else:
                    st.error("❌ " + msg)

                # Show completed combinations (from RAW data)
                completed = dashboard_utils.get_completed_experiments(raw_df, required_repeats)
                if completed:
                    with st.expander("Completed Combinations", expanded=True):
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
    st.header("Plots")
    try:
        if df is not None and not df.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Success Rate per Perturbation")
                df['clean_pert'] = df['perturbation'].apply(lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x))

                if 'binary_SR' in df.columns:
                    pert_sr = df.groupby('clean_pert')['binary_SR'].mean().reset_index()

                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(pert_sr['clean_pert'], pert_sr['binary_SR'], color='skyblue')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Success Rate")
                    ax.set_xlabel("Perturbation")
                    plt.xticks(rotation=45, ha='right')

                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    st.image(buf)
                else:
                    st.info("No binary_SR column found for plots.")

            with c2:
                st.subheader("Success Rate per Task")
                if 'binary_SR' in df.columns:
                    task_sr = df.groupby('task')['binary_SR'].mean().reset_index()

                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.bar(task_sr['task'], task_sr['binary_SR'], color='lightgreen')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Success Rate")
                    ax.set_xlabel("Task")
                    plt.xticks(rotation=45, ha='right')

                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    st.image(buf)
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
