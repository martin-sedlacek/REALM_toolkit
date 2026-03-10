import os
import glob
import pandas as pd
import json

SUPPORTED_TASKS = [
    "put_green_block_in_bowl", #0
    "put_banana_into_box", #1
    "rotate_marker", #2
    "rotate_mug", #3
    "pick_spoon", #4
    "pick_water_bottle", #5
    "stack_cubes", #6
    "push_switch", #7
    "open_drawer", #8
    "close_drawer", #9
]

SUPPORTED_PERTURBATIONS = [
    'Default', #0
    'V-AUG', # 1
    'V-VIEW', # 2
    'V-SC', # 3
    'V-LIGHT', # 4
    'S-PROP', # 5
    'S-LANG', # 6
    'S-MO', # 7
    'S-AFF', # 8
    'S-INT', # 9
    'B-HOBJ', # 10
    'SB-NOUN', # 11
    'SB-VRB', # 12
    'VB-POSE', # 13
    'VB-MOBJ', # 14
    'VSB-NOBJ' # 15
]

def get_subdirectories(path):
    if not os.path.exists(path):
        return []
    try:
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    except OSError:
        return []

def is_experiment_folder(path):
    """Check if the folder contains 'reports' or 'videos' subdirectories."""
    return os.path.isdir(os.path.join(path, "reports")) or os.path.isdir(os.path.join(path, "videos"))

def load_reports(experiment_path):
    reports_path = os.path.join(experiment_path, "reports")
    if not os.path.exists(reports_path):
        return None

    csv_files = glob.glob(os.path.join(reports_path, "*.csv"))
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            # We can't log to streamlit here directly without passing st, but we can print or ignore
            print(f"Error reading {f}: {e}")

    if not dfs:
        return None

    try:
        aggregated_df = pd.concat(dfs, axis=0, ignore_index=True)
        return aggregated_df
    except Exception as e:
        print(f"Error aggregating CSVs: {e}")
        return None

def get_videos(experiment_path):
    videos_path = os.path.join(experiment_path, "videos")
    if not os.path.exists(videos_path):
        return []
    return sorted(glob.glob(os.path.join(videos_path, "*.mp4")))

def load_experiment_metadata(experiment_path):
    """Loads metadata.json from the experiment directory."""
    metadata_path = os.path.join(experiment_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None, f"Metadata file not found at {metadata_path}"

    try:
        import re
        with open(metadata_path, 'r') as f:
            content = f.read()
            
        def repl(match):
            start = int(match.group(1))
            end = int(match.group(2))
            return "[" + ", ".join(str(i) for i in range(start, end + 1)) + "]"
        
        content = re.sub(r'\[\s*(\d+)\s*-\s*(\d+)\s*\]', repl, content)
        metadata = json.loads(content)
        return metadata, None
    except Exception as e:
        return None, f"Error reading metadata: {e}"

def check_experiment_status(df, tasks_indices, perts_indices, required_repeats):
    if df is None:
        return False, "No data loaded."

    target_tasks = []
    for i in tasks_indices:
        if 0 <= i < len(SUPPORTED_TASKS):
            target_tasks.append(SUPPORTED_TASKS[i])

    target_perts = []
    for i in perts_indices:
        if 0 <= i < len(SUPPORTED_PERTURBATIONS):
            # Format to match CSV string representation of list
            target_perts.append(f"{SUPPORTED_PERTURBATIONS[i]}")

    if not target_tasks or not target_perts:
        return False, "Invalid task or perturbation indices in experiment name."

    # Check existence
    all_good = True
    missing = []

    for t in target_tasks:
        for p in target_perts:
            # Filter df
            count = len(df[(df['task'] == t) & (df['perturbation'] == p)])
            if count < required_repeats:
                all_good = False
                missing.append(f"{t} | {p}: Found {count}/{required_repeats}")

    if all_good:
        return True, "All required tasks and perturbations evaluated with sufficient samples."
    else:
        return False, "Missing evaluations:\n" + "\n".join(missing)

def get_completed_experiments(df, required_repeats=1):
    """
    Returns a list of tuples (task, perturbation) that have >= required_repeats in the dataframe.
    """
    if df is None or df.empty:
        return []

    completed = []
    try:
        counts = df.groupby(['task', 'perturbation']).size()

        for (task, pert_str), count in counts.items():
            if count >= required_repeats:
                pert_clean = pert_str.replace("['", "").replace("']", "")
                completed.append((task, pert_clean))
        completed.sort()
    except Exception as e:
        print(f"Error in get_completed_experiments: {e}")
        return []

    return completed

def filter_dataframe(df, selected_tasks=None, selected_perts=None):
    """
    Filters the dataframe based on selected tasks and perturbations.
    """
    if df is None or df.empty:
        return df

    filtered_df = df.copy()

    # 1. Clean perturbation column to match simplified names
    # The dataframe 'perturbation' column often looks like "['Default']"
    # We create a temporary column for filtering
    filtered_df['_pert_clean'] = filtered_df['perturbation'].apply(
        lambda x: x.replace("['", "").replace("']", "") if isinstance(x, str) else str(x)
    )

    if selected_tasks:
        filtered_df = filtered_df[filtered_df['task'].isin(selected_tasks)]

    if selected_perts:
        filtered_df = filtered_df[filtered_df['_pert_clean'].isin(selected_perts)]

    # Clean up temp column
    filtered_df = filtered_df.drop(columns=['_pert_clean'])

    return filtered_df

def filter_videos(video_paths, selected_tasks=None, selected_perts=None):
    """
    Filters video paths based on selected tasks and perturbations lists.
    Logic: (Task in selected_tasks OR empty) AND (Pert in selected_perts OR empty).
    """
    if not selected_tasks and not selected_perts:
        return video_paths

    filtered = []
    for v in video_paths:
        filename = os.path.basename(v)

        # Check Task
        task_match = False
        if not selected_tasks:
            task_match = True
        else:
            for t in selected_tasks:
                if t in filename:
                    task_match = True
                    break

        if not task_match:
            continue

        # Check Perturbation
        pert_match = False
        if not selected_perts:
            pert_match = True
        else:
            for p in selected_perts:
                # We need to be careful with substring matching for perturbations.
                # e.g., 'V-AUG' in filename.
                # However, filename format is typically: ..._rollout_TASK_PERT_...
                # So the pert string is surrounded by underscores or end of string?
                # But simple inclusion is likely sufficient given the uniqueness of names in SUPPORTED_PERTURBATIONS.
                if p in filename:
                    pert_match = True
                    break

        if pert_match:
            filtered.append(v)

    return filtered
