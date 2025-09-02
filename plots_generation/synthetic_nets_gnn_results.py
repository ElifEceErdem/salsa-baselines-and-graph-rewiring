import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# --- File Paths and Helper Functions ---
# Define the path to the network properties file.
prop_path = "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/results/networks_info.csv"
# Define the output directory for the generated plots.
output_plot_path = "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/plots/"

def base_network(net_name):
    """
    Extracts the base network name from a versioned network name.
    Example: 'gnn_synthetic_network1_v1' -> 'gnn_synthetic_network1'
    """
    # This regex is updated to match the GNN network naming convention.
    m = re.match(r'(gnn_synthetic_network\d+)_v\d+', str(net_name))
    if m:
        return m.group(1)
    return net_name

def load_and_tag(path):
    """Loads a CSV and adds a 'base_net' column for easier grouping."""
    df = pd.read_csv(path)
    df['base_net'] = df['network'].apply(base_network)
    return df

# --- Data Loading ---
# Load the results from different models.
salsa_results = load_and_tag("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/results/salsa_all_results.csv")
ssnet_results = load_and_tag("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/results/ssnet_all_results.csv")
tailsteak_results = load_and_tag("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/results/tailsteak_all_results.csv")

def avg_metric(df, netlist, metric_col, **query):
    """
    Calculates the average of a given metric for a list of networks.
    Filters by additional query parameters if provided.
    """
    vals = []
    for net in netlist:
        # Start with the base network name match.
        mask = (df['base_net'] == net)
        # Apply additional filters from the query.
        for k, v in query.items():
            mask &= (df.get(k) == v)
        rows = df[mask]
        # Calculate mean, or return NaN if no data matches.
        vals.append(rows[metric_col].mean() if not rows.empty else np.nan)
    return vals

def extract_float(s):
    """Extracts the first float from a string, e.g., '0.007 (0.000)' -> 0.007."""
    if pd.isnull(s):
        return np.nan
    return float(str(s).split()[0])

def to_float_safe(val):
    """Safely converts a value to float, returning NaN on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

# Load and preprocess network properties.
gnn_properties = pd.read_csv(prop_path).set_index("network")
gnn_properties['clustering'] = gnn_properties['clustering'].apply(extract_float)
gnn_properties['density'] = gnn_properties['density'].apply(extract_float)


# ==============================================================================
# /// PART 1: ANALYSIS FOR CLUSTERING COMPARISON ///
# ==============================================================================
print("--- Starting Part 1: GNN Clustering Analysis ---")

# --- Define the GNN-based networks ---
sparse_nets = ['gnn_synthetic_network1', 'gnn_synthetic_network2', 'gnn_synthetic_network3', 'gnn_synthetic_network4']
dense_nets  = ['gnn_synthetic_network5', 'gnn_synthetic_network6', 'gnn_synthetic_network7', 'gnn_synthetic_network8']

# --- Extract clustering coefficients for the GNN networks ---
clust_sparse = gnn_properties.loc[sparse_nets, "clustering"].values
clust_dense  = gnn_properties.loc[dense_nets,  "clustering"].values

# --- Prepare metrics for the GNN networks ---
# NOTE: The incorrect 'model' filters for SALSA and TailSTEAK have been removed.
lgn_prec_sparse = avg_metric(ssnet_results, sparse_nets, "p10", model="LightGCN", mode="normal")
lgn_fair_sparse = avg_metric(ssnet_results, sparse_nets, "1-gini", model="LightGCN", mode="normal")
lgn_prec_dense  = avg_metric(ssnet_results, dense_nets, "p10", model="LightGCN", mode="normal")
lgn_fair_dense  = avg_metric(ssnet_results, dense_nets, "1-gini", model="LightGCN", mode="normal")

pprgo_prec_sparse = avg_metric(ssnet_results, sparse_nets, "p10", model="PPRGo", mode="normal")
pprgo_fair_sparse = avg_metric(ssnet_results, sparse_nets, "1-gini", model="PPRGo", mode="normal")
pprgo_prec_dense  = avg_metric(ssnet_results, dense_nets, "p10", model="PPRGo", mode="normal")
pprgo_fair_dense  = avg_metric(ssnet_results, dense_nets, "1-gini", model="PPRGo", mode="normal")

salsa_prec_sparse = avg_metric(salsa_results, sparse_nets, "precision")
salsa_fair_sparse = avg_metric(salsa_results, sparse_nets, "1-gini index")
salsa_prec_dense  = avg_metric(salsa_results, dense_nets, "precision")
salsa_fair_dense  = avg_metric(salsa_results, dense_nets, "1-gini index")

tailsteak_prec_sparse = avg_metric(tailsteak_results, sparse_nets, "p@5")
tailsteak_fair_sparse = avg_metric(tailsteak_results, sparse_nets, "1_minus_gini")
tailsteak_prec_dense  = avg_metric(tailsteak_results, dense_nets, "p@5")
tailsteak_fair_dense  = avg_metric(tailsteak_results, dense_nets, "1_minus_gini")

# --- Generate the 2x2 Plotting vs. Clustering ---
label_fontsize = 17
tick_fontsize = 13

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='row')

# Top-left: Sparse Precision
ax = axes[0, 0]
ax.plot(clust_sparse, salsa_prec_sparse, 'bo-', label='SALSA')
ax.plot(clust_sparse, lgn_prec_sparse, 'ro-', label='LightGCN')
ax.plot(clust_sparse, pprgo_prec_sparse, 'go-', label='PPRGo')
ax.plot(clust_sparse, tailsteak_prec_sparse, 'mo-', label='TailSTEAK')
ax.set_ylabel("Precision @ 10", fontsize=label_fontsize)
ax.set_title("Sparse Networks", fontsize=label_fontsize)
ax.legend()
ax.tick_params(axis='y', labelsize=tick_fontsize)

# Top-right: Dense Precision
ax = axes[0, 1]
ax.plot(clust_dense, salsa_prec_dense, 'bo-', label='SALSA')
ax.plot(clust_dense, lgn_prec_dense, 'ro-', label='LightGCN')
ax.plot(clust_dense, pprgo_prec_dense, 'go-', label='PPRGo')
ax.plot(clust_dense, tailsteak_prec_dense, 'mo-', label='TailSTEAK')
ax.set_title("Dense Networks",fontsize=label_fontsize)
ax.legend()

# Bottom-left: Sparse Fairness
ax = axes[1, 0]
ax.plot(clust_sparse, salsa_fair_sparse, 'bo-', label='SALSA')
ax.plot(clust_sparse, lgn_fair_sparse, 'ro-', label='LightGCN')
ax.plot(clust_sparse, pprgo_fair_sparse, 'go-', label='PPRGo')
ax.plot(clust_sparse, tailsteak_fair_sparse, 'mo-', label='TailSTEAK')
ax.set_xlabel("Clustering Coefficient", fontsize=label_fontsize)
ax.set_ylabel("Fairness (1-Gini)", fontsize=label_fontsize)
ax.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
ax.tick_params(axis='y', labelsize=tick_fontsize)
ax.legend()

# Bottom-right: Dense Fairness
ax = axes[1, 1]
ax.plot(clust_dense, salsa_fair_dense, 'bo-', label='SALSA')
ax.plot(clust_dense, lgn_fair_dense, 'ro-', label='LightGCN')
ax.plot(clust_dense, pprgo_fair_dense, 'go-', label='PPRGo')
ax.plot(clust_dense, tailsteak_fair_dense, 'mo-', label='TailSTEAK')
ax.set_xlabel("Clustering Coefficient", fontsize=label_fontsize)
ax.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{output_plot_path}clustering_vs_metrics_2x2_gnn_based.pdf")
plt.close()
print("Clustering plots generated for GNN-based networks.")


# =============================================================================
# /// PART 2: ANALYSIS FOR DENSITY COMPARISON (Matched Pairs) ///
# =============================================================================
print("\n--- Starting Part 2: GNN Density Analysis (4 matched pairs) ---")

# --- Define the 4 matched network pairs ---
pairs_density = [
    ('gnn_synthetic_network1', 'gnn_synthetic_network5'),
    ('gnn_synthetic_network2', 'gnn_synthetic_network6'),
    ('gnn_synthetic_network3', 'gnn_synthetic_network7'),
    ('gnn_synthetic_network4', 'gnn_synthetic_network8'),
]
num_pairs_density = len(pairs_density)
sparse_nets_density = [p[0] for p in pairs_density]
dense_nets_density  = [p[1] for p in pairs_density]

# --- Prepare data for Paired Density Plots ---
densities_sparse = gnn_properties.loc[sparse_nets_density, "density"].values
densities_dense  = gnn_properties.loc[dense_nets_density,  "density"].values

# --- Recalculate metrics for the 4 matched pairs ---
lgn_prec_sparse_density = avg_metric(ssnet_results, sparse_nets_density, "p10", model="LightGCN", mode="normal")
lgn_fair_sparse_density = avg_metric(ssnet_results, sparse_nets_density, "1-gini", model="LightGCN", mode="normal")
lgn_prec_dense_density  = avg_metric(ssnet_results, dense_nets_density, "p10", model="LightGCN", mode="normal")
lgn_fair_dense_density  = avg_metric(ssnet_results, dense_nets_density, "1-gini", model="LightGCN", mode="normal")

pprgo_prec_sparse_density = avg_metric(ssnet_results, sparse_nets_density, "p10", model="PPRGo", mode="normal")
pprgo_fair_sparse_density = avg_metric(ssnet_results, sparse_nets_density, "1-gini", model="PPRGo", mode="normal")
pprgo_prec_dense_density  = avg_metric(ssnet_results, dense_nets_density, "p10", model="PPRGo", mode="normal")
pprgo_fair_dense_density  = avg_metric(ssnet_results, dense_nets_density, "1-gini", model="PPRGo", mode="normal")

salsa_prec_sparse_density = avg_metric(salsa_results, sparse_nets_density, "precision")
salsa_fair_sparse_density = avg_metric(salsa_results, sparse_nets_density, "1-gini index")
salsa_prec_dense_density  = avg_metric(salsa_results, dense_nets_density, "precision")
salsa_fair_dense_density  = avg_metric(salsa_results, dense_nets_density, "1-gini index")

tailsteak_prec_sparse_density = avg_metric(tailsteak_results, sparse_nets_density, "p@5")
tailsteak_fair_sparse_density = avg_metric(tailsteak_results, sparse_nets_density, "1_minus_gini")
tailsteak_prec_dense_density  = avg_metric(tailsteak_results, dense_nets_density, "p@5")
tailsteak_fair_dense_density  = avg_metric(tailsteak_results, dense_nets_density, "1_minus_gini")

models_density = [
    ("SALSA", salsa_prec_sparse_density, salsa_prec_dense_density, salsa_fair_sparse_density, salsa_fair_dense_density),
    ("LightGCN", lgn_prec_sparse_density, lgn_prec_dense_density, lgn_fair_sparse_density, lgn_fair_dense_density),
    ("PPRGo", pprgo_prec_sparse_density, pprgo_prec_dense_density, pprgo_fair_sparse_density, pprgo_fair_dense_density),
    ("TailSTEAK", tailsteak_prec_sparse_density, tailsteak_prec_dense_density, tailsteak_fair_sparse_density, tailsteak_fair_dense_density),
]
model_colors = {'SALSA': 'b', 'LightGCN': 'r', 'PPRGo': 'g', 'TailSTEAK': 'm'}

# --- Combined Delta Bar Plot for Precision and Fairness ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
x = np.arange(num_pairs_density)
bar_width = 0.2

# Top Plot: Delta Precision
ax_prec = axes[0]
for i, (name, prec_sparse, prec_dense, _, _) in enumerate(models_density):
    deltas = np.array([to_float_safe(d) for d in prec_dense]) - np.array([to_float_safe(s) for s in prec_sparse])
    ax_prec.bar(x + i * bar_width, deltas, width=bar_width, label=name, color=model_colors[name])
ax_prec.set_ylabel('Δ Precision (Dense - Sparse)', fontsize=17)
ax_prec.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax_prec.legend()
ax_prec.tick_params(axis='y', labelsize=15)

# Bottom Plot: Delta Fairness
ax_fair = axes[1]
for i, (name, _, _, fair_sparse, fair_dense) in enumerate(models_density):
    deltas = np.array([to_float_safe(d) for d in fair_dense]) - np.array([to_float_safe(s) for s in fair_sparse])
    ax_fair.bar(x + i * bar_width, deltas, width=bar_width, label=name, color=model_colors[name])

xtick_labels = [f'Net{p[0].split("network")[1]}→Net{p[1].split("network")[1]}' for p in pairs_density]
ax_fair.set_xticks(x + 1.5 * bar_width)
ax_fair.set_xticklabels(xtick_labels, fontsize=16)
ax_fair.set_ylabel('Δ Fairness (Dense - Sparse)', fontsize=17)
ax_fair.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax_fair.legend()
ax_fair.tick_params(axis='y', labelsize=15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_plot_path}delta_barplot_combined_gnn_based.pdf", bbox_inches="tight")
plt.close()

print("Combined density delta plot for GNN-based networks generated.")
print("\n✅ All GNN analyses complete.")

