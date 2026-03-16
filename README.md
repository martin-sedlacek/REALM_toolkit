# REALM Toolkit

A Streamlit dashboard for browsing and visualizing [REALM](https://github.com/martin-sedlacek/REALM) experiment results.

## Features

*   **Experiment Browser**: A tree-style browser for navigating through experiment logs.
*   **Success Rate Plots**: Visualize success rates per task and per perturbation.
*   **Task Progression Plots**: Visualize task progression per task.
*   **Stage Frequency Plots**: Visualize the frequency of different stages, ordered by task progression.
*   **Video Gallery**: View and filter videos from experiments.
*   **Aggregated Reports**: View and filter aggregated experiment data.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then create the environment:

```bash
uv sync
```

## Configuration

Set the `REALM_LOGS` environment variable to the `logs/` folder inside your REALM installation:

```bash
export REALM_LOGS=/path/to/REALM/logs
```

For example, if REALM is installed at `~/projects/REALM`, set:

```bash
export REALM_LOGS=~/projects/REALM/logs
```

You can add this to your shell profile (`~/.zshrc`, `~/.bashrc`) to make it permanent.

## Running the Dashboard

```bash
uv run streamlit run realm_viewer/dashboard.py
```

Or inline with the env variable:

```bash
REALM_LOGS=/path/to/REALM/logs uv run streamlit run realm_viewer/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`.
