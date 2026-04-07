# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REALM_toolkit is a Streamlit-based experiment visualization dashboard for browsing, filtering, and analyzing robotic manipulation experiment results stored in a hierarchical `logs/` directory.

## Environment

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                   # create/update .venv and install deps
uv run streamlit run realm_viewer/dashboard.py   # run the dashboard
uv run python -m unittest realm_viewer.test_dashboard_utils          # run tests
uv run python -m unittest realm_viewer.test_dashboard_utils.TestDashboardUtils.test_filter_videos  # single test
```

Set `REALM_LOGS` to override the default `logs/` directory:

```bash
REALM_LOGS=/path/to/logs uv run streamlit run realm_viewer/dashboard.py
```

## Architecture

**Two-module design:**
- `realm_viewer/dashboard.py` — Streamlit UI: sidebar directory tree, filters, status/plots/reports/video tabs, session state management
- `realm_viewer/dashboard_utils.py` — Pure utility functions (no Streamlit imports): file I/O, data loading, filtering, experiment status checking

**Constants** in `dashboard_utils.py` define the domain:
- `SUPPORTED_TASKS`: 10 robotic manipulation task names
- `SUPPORTED_PERTURBATIONS`: 16 perturbation types (Default, V-AUG, V-VIEW, S-LANG, etc.)

**Expected runtime directory structure:**
```
logs/
└── [experiment_name]/
    └── [model_name]/
        └── [run_id]/
            ├── metadata.json    # experiment config with range notation support
            ├── reports/         # *.csv result files
            └── videos/          # *.mp4 rollout videos
```

**Data flow:** User navigates sidebar tree → selects experiment folder → utils load reports (CSVs aggregated via pandas), metadata (JSON), videos (MP4 paths) → filters applied → Streamlit renders charts (matplotlib), dataframes, and paginated video gallery.

**Key Streamlit patterns used:**
- `@st.cache_data` for caching loaded data
- `@st.fragment` for partial reruns (video section)
- `st.session_state` for selected experiment, expanded folders, pagination

## Testing

Tests use Python's `unittest` framework and test utility functions in isolation (no Streamlit). Tests are in `realm_viewer/test_dashboard_utils.py`.
