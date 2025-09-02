import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# TODO: Please verify this path to the original graph is correct.
REED98_GRAPH_FILE = "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/tailsteak_fairness_examination/socfb-Reed98.edgelist"

DATASET_CONFIG = [
    {
        "name": "PPRGo (Reed98)",
        "original_graph_file": REED98_GRAPH_FILE,
        "predicted_edges_file": "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/tailsteak_fairness_examination/pprgo_predicted_network_top10.npy"
    },
    {
        "name": "Tailsteak (Reed98)",
        "original_graph_file": REED98_GRAPH_FILE,
        "predicted_edges_file": "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/tailsteak_fairness_examination/tailsteak_top_10_predicted_edges_seed_7.txt"
    }
]

NUM_BINS = 12
# Saving output to a local 'output' directory for convenience
OUTPUT_DIR = "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/tailsteak_fairness_examination/"
OUTPUT_FILENAME = "degree_mixing_comparison_pprgo_vs_tailsteak.png"
# ---------------------

def prepare_data_for_dataset(config):
    """
    Loads and processes data for a single dataset.
    This function now handles both .txt and .npy prediction files.
    Returns a dictionary with the normalized matrix and labels, or None on failure.
    """
    dataset_name = config["name"]
    print(f"\n--- Processing Data For: {dataset_name} ---")

    try:
        # Load original graph to get initial degrees
        G_orig = nx.read_edgelist(config["original_graph_file"], delimiter=None, nodetype=int)
        initial_degrees = dict(G_orig.degree())
        if not initial_degrees: raise ValueError("Graph is empty or could not be read.")

        # Load predicted edges based on file type
        prediction_file = config["predicted_edges_file"]
        if prediction_file.endswith('.npy'):
            print(f"Reading NumPy file: {os.path.basename(prediction_file)}")
            # predicted_edges = np.load(prediction_file)
            predicted_edges = np.load(prediction_file)[:, :2]
        elif prediction_file.endswith('.txt'):
            print(f"Reading text file: {os.path.basename(prediction_file)}")
            # Assuming format: node1,node2 (or node1 node2) per line
            # The split() method handles any whitespace delimiter, which is robust.
            # If your file is strictly comma-separated, you can use split(',')
            predicted_edges = [tuple(map(int, line.strip().split(','))) for line in open(prediction_file)]
        else:
            raise ValueError(f"Unsupported file format for predictions: {prediction_file}")

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error processing '{dataset_name}': {e}. Skipping.")
        return None
    
    max_degree = max(initial_degrees.values()) if initial_degrees else 1
    bins = np.unique(np.logspace(0, np.log2(max_degree if max_degree > 0 else 1), num=NUM_BINS, base=2, dtype=int)).tolist()
    if not bins: bins = [1]
    if bins[0] != 1: bins.insert(0, 1)

    mixing_matrix = np.zeros((len(bins), len(bins)))
    for source, target in predicted_edges:
        if source in initial_degrees and target in initial_degrees:
            source_bin = np.searchsorted(bins, initial_degrees[source], side='right') - 1
            target_bin = np.searchsorted(bins, initial_degrees[target], side='right') - 1
            if source_bin >= 0 and target_bin >= 0:
                mixing_matrix[source_bin, target_bin] += 1
    
    row_sums = mixing_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(mixing_matrix, row_sums, where=row_sums!=0)
    
    bin_labels = [f"{bins[i]}-{bins[i+1]-1}" if i+1 < len(bins) and bins[i] < bins[i+1]-1 else f"{bins[i]}" for i in range(len(bins))]
        
    return {"matrix": normalized_matrix, "labels": bin_labels, "name": dataset_name}



from matplotlib.ticker import MaxNLocator # Import the locator
import matplotlib.gridspec as gridspec

def main_side_by_side():
    # =========================================================================
    # CRITICAL STEP 1: First, prepare data for ALL datasets to find the global scale.
    # =========================================================================
    plot_data_list = [prepare_data_for_dataset(config) for config in DATASET_CONFIG]
    plot_data_list = [data for data in plot_data_list if data is not None]

    if not plot_data_list:
        print("No valid data could be processed. Exiting.")
        return

    # Find the single maximum value across all processed matrices.
    global_vmax = max(data['matrix'].max() for data in plot_data_list if data['matrix'].size > 0)
    print(f"\n✅ Global color scale determined. Max value (vmax) = {global_vmax:.3f}")

    # =========================================================================
    # CRITICAL STEP 2: Now, create the plot, forcing each heatmap to use the same scale.
    # =========================================================================
    print("Generating plot...")

    # Make a figure with one extra column reserved for the shared colorbar
    fig = plt.figure(figsize=(11 * len(plot_data_list), 8), constrained_layout=True)
    gs = fig.add_gridspec(
        1, len(plot_data_list) + 1,
        width_ratios=[1] * len(plot_data_list) + [0.04]  # last column is the colorbar
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(len(plot_data_list))]
    cax = fig.add_subplot(gs[0, -1])  # dedicated colorbar axis

    mappable_for_cbar = None
    for i, data in enumerate(plot_data_list):
        ax = axes[i]
        hm = sns.heatmap(
            data['matrix'],
            cmap='inferno',
            linewidths=.5,
            ax=ax,
            xticklabels=data['labels'],
            yticklabels=data['labels'],
            cbar=False,           # IMPORTANT: no per-axis colorbar
            vmin=0,
            vmax=global_vmax
        )
        # remember a mappable for the shared colorbar
        if mappable_for_cbar is None:
            mappable_for_cbar = hm.collections[0]

        ax.set_title(f"{data['name']}", fontsize=24, pad=10)
        ax.set_xlabel('Target Node Degree Bin', fontsize=22)
        ax.tick_params(axis='x', rotation=45, labelsize=18)
        ax.tick_params(axis='y', rotation=0, labelsize=18)

    # Y-label on the first heatmap only
    axes[0].set_ylabel('Source Node Degree Bin', fontsize=22)

    # Shared colorbar placed in its own axis -> no overlap
    cbar = fig.colorbar(mappable_for_cbar, cax=cax, orientation='vertical')
    cbar.set_label('Normalized Frequency', fontsize=22, labelpad=15)
    cbar.ax.tick_params(labelsize=18)

    # plt.show()

    # (If you want to save)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(full_output_path, dpi=300)
    plt.close(fig)


def main():
    # STEP 1: Prepare data and find global scale
    plot_data_list = [prepare_data_for_dataset(config) for config in DATASET_CONFIG]
    plot_data_list = [data for data in plot_data_list if data is not None]

    if not plot_data_list:
        print("No valid data to be plotted.")
        return

    global_vmax = max(data['matrix'].max() for data in plot_data_list if data['matrix'].size > 0)
    print(f"\n✅ Global color scale determined. Max value (vmax) = {global_vmax:.3f}")

    # =========================================================================
    # STEP 2: Create the plot with updated titles and shared axes
    # =========================================================================
    print("Generating plot...")

    # === KEY CHANGE 1: Use constrained_layout for better automatic spacing ===
    fig = plt.figure(figsize=(8, 11), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.1)

    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    cax = fig.add_subplot(gs[:, 1])

    mappable_for_cbar = None
    subplot_labels = ['(a)', '(b)']

    for i, data in enumerate(plot_data_list):
        ax = axes[i]
        hm = sns.heatmap(
            data['matrix'], cmap='inferno', linewidths=.5, ax=ax,
            xticklabels=data['labels'], yticklabels=data['labels'],
            cbar=False, vmin=0, vmax=global_vmax
        )
        if mappable_for_cbar is None:
            mappable_for_cbar = hm.collections[0]
        
        ax.set_title(f"{subplot_labels[i]} {data['name']}", fontsize=22, pad=15)
        ax.set_ylabel('Source Node Degree Bin', fontsize=22)
        ax.tick_params(axis='y', rotation=0, labelsize=18)

        if i == len(plot_data_list) - 1:
            ax.set_xlabel('Target Node Degree Bin', fontsize=22, labelpad=15)
            ax.tick_params(axis='x', rotation=45, labelsize=18)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])

    cbar = fig.colorbar(mappable_for_cbar, cax=cax, orientation='vertical')
    cbar.set_label('Normalized Frequency', fontsize=22, labelpad=15)
    cbar.ax.tick_params(labelsize=18)

    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # === KEY CHANGE 2: Corrected the function name to os.path.join ===
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Plot successfully saved as '{full_output_path}'")


if __name__ == '__main__':
    main()
