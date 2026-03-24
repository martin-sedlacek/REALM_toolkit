import matplotlib.pyplot as plt
import streamlit as st


def plot_model_legend(ax, model_to_marker, model_to_color):
    """Core logic to plot the model legend on a given ax."""
    if not model_to_marker:
        return

    unique_models = sorted(model_to_marker.keys())
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax.axis('off')

    # Use two handles for each model: a patch for the bar/violin color and a Line2D for the marker shape
    legend_elements = []
    for m in unique_models:
        color = model_to_color[m]
        marker = model_to_marker[m]
        legend_elements.append((Patch(facecolor=color, edgecolor='black', alpha=0.8),
                               Line2D([0], [0], marker=marker, color='w', markerfacecolor='black',
                                      markeredgecolor='black', markersize=10)))

    from matplotlib.legend_handler import HandlerTuple
    ax.legend(handles=legend_elements, labels=unique_models, loc='center', ncol=min(len(unique_models), 5),
              title="Models", frameon=False, fontsize=10, handler_map={tuple: HandlerTuple(ndivide=None)})


def render_model_legend(model_to_marker, model_to_color):
    """Render a standalone legend infographic for models."""
    fig, ax = plt.subplots(figsize=(10, 0.8))
    plot_model_legend(ax, model_to_marker, model_to_color)
    st.pyplot(fig, transparent=True)
    plt.close(fig)
