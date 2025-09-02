import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# --- File Paths and Helper Functions (No changes here) ---
prop_path = "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/results/networks_info.csv"

def base_network(net_name):
    m = re.match(r'(edge_swap_net\d+)_v\d+', str(net_name))
    if m:
        return m.group(1)
    return net_name

def load_and_tag(path):
    df = pd.read_csv(path)
    df['base_net'] = df['network'].apply(base_network)
    return df

salsa_results = load_and_tag("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/results/salsa_all_results.csv")
ssnet_results = load_and_tag("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/results/ssnet_all_results.csv")
tailsteak_results = load_and_tag("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/results/tailsteak_all_results.csv")

def avg_metric(df, netlist, metric_col, **query):
    vals = []
    for net in netlist:
        mask = (df['base_net'] == net)
        for k, v in query.items():
            mask &= (df.get(k, None) == v)
        rows = df[mask]
        vals.append(rows[metric_col].mean() if not rows.empty else np.nan)
    return vals

def extract_float(s):
    if pd.isnull(s):
        return np.nan
    return float(str(s).split()[0])

def to_float_safe(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

gnn_properties = pd.read_csv(prop_path).set_index("network")
gnn_properties['clustering'] = gnn_properties['clustering'].apply(extract_float)

# ==============================================================================
# /// PART 1: ANALYSIS FOR CLUSTERING COMPARISON (11 Systematic Pairs) ///
# ==============================================================================
print("--- Starting Part 1: Clustering Analysis (11 pairs) ---")

# --- Define networks to exclude from the analysis ---
nets_to_exclude = ['edge_swap_net5']
print(f"Excluding the following networks from analysis: {nets_to_exclude}")

# --- Define the 11 systematic network pairs, filtering out the excluded ones ---
num_pairs_clust = 11
sparse_nets_clust = [f'edge_swap_net{i}' for i in range(1, num_pairs_clust + 1) if f'edge_swap_net{i}' not in nets_to_exclude]
dense_nets_clust  = [f'edge_swap_net{i}' for i in range(num_pairs_clust + 1, 2 * num_pairs_clust + 1)]


# --- Extract clustering coefficients for the 11 pairs ---
clust_sparse = gnn_properties.loc[sparse_nets_clust, "clustering"].values
clust_dense  = gnn_properties.loc[dense_nets_clust,  "clustering"].values

# --- Prepare metrics for the 11 pairs ---
lgn_prec_sparse_clust = avg_metric(ssnet_results, sparse_nets_clust, "p10", model="LightGCN", mode="normal")
lgn_fair_sparse_clust = avg_metric(ssnet_results, sparse_nets_clust, "1-gini", model="LightGCN", mode="normal")
lgn_prec_dense_clust  = avg_metric(ssnet_results, dense_nets_clust, "p10", model="LightGCN", mode="normal")
lgn_fair_dense_clust  = avg_metric(ssnet_results, dense_nets_clust, "1-gini", model="LightGCN", mode="normal")
pprgo_prec_sparse_clust = avg_metric(ssnet_results, sparse_nets_clust, "p10", model="PPRGo", mode="normal")
pprgo_fair_sparse_clust = avg_metric(ssnet_results, sparse_nets_clust, "1-gini", model="PPRGo", mode="normal")
pprgo_prec_dense_clust  = avg_metric(ssnet_results, dense_nets_clust, "p10", model="PPRGo", mode="normal")
pprgo_fair_dense_clust  = avg_metric(ssnet_results, dense_nets_clust, "1-gini", model="PPRGo", mode="normal")
tailsteak_prec_sparse_clust = avg_metric(tailsteak_results, sparse_nets_clust, "p@5")
tailsteak_fair_sparse_clust = avg_metric(tailsteak_results, sparse_nets_clust, "1_minus_gini")
tailsteak_prec_dense_clust  = avg_metric(tailsteak_results, dense_nets_clust, "p@5")
tailsteak_fair_dense_clust  = avg_metric(tailsteak_results, dense_nets_clust, "1_minus_gini")

print("XXXX")
salsa_prec_sparse_clust = avg_metric(salsa_results, sparse_nets_clust, "precision")
salsa_fair_sparse_clust = avg_metric(salsa_results, sparse_nets_clust, "1-gini index")
salsa_prec_dense_clust  = avg_metric(salsa_results, dense_nets_clust, "precision")
salsa_fair_dense_clust  = avg_metric(salsa_results, dense_nets_clust, "1-gini index")


# --- MODIFIED PLOTTING SECTION ---
# --- Generate the 2x2 Plotting vs. Clustering ---
# Define font sizes for consistency
label_fontsize = 17
tick_fontsize = 13

# (3) sharey='row' added to keep y-axis scales consistent across rows
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='row')
# --- Top-left: Sparse Precision ---
ax = axes[0, 0]
ax.plot(clust_sparse, salsa_prec_sparse_clust, 'bo-', label='SALSA')
ax.plot(clust_sparse, lgn_prec_sparse_clust, 'ro-', label='LightGCN')
ax.plot(clust_sparse, pprgo_prec_sparse_clust, 'go-', label='PPRGo')
ax.plot(clust_sparse, tailsteak_prec_sparse_clust, 'mo-', label='TailSTEAK')
# (1) Increased font size for y-axis label
ax.set_ylabel("Precision @ 10", fontsize=label_fontsize)
ax.set_title("Sparse Networks", fontsize=label_fontsize)
ax.legend()
# (2) Increased font size for y-axis ticks
ax.tick_params(axis='y', labelsize=tick_fontsize)

# --- Top-right: Dense Precision ---
ax = axes[0, 1]
ax.plot(clust_dense, salsa_prec_dense_clust, 'bo-', label='SALSA')
ax.plot(clust_dense, lgn_prec_dense_clust, 'ro-', label='LightGCN')
ax.plot(clust_dense, pprgo_prec_dense_clust, 'go-', label='PPRGo')
ax.plot(clust_dense, tailsteak_prec_dense_clust, 'mo-', label='TailSTEAK')
ax.set_title("Dense Networks",fontsize=label_fontsize)
ax.legend()
# y-axis label and ticks are shared from the left plot due to sharey='row'

# --- Bottom-left: Sparse Fairness ---
ax = axes[1, 0]
ax.plot(clust_sparse, salsa_fair_sparse_clust, 'bo-', label='SALSA')
ax.plot(clust_sparse, lgn_fair_sparse_clust, 'ro-', label='LightGCN')
ax.plot(clust_sparse, pprgo_fair_sparse_clust, 'go-', label='PPRGo')
ax.plot(clust_sparse, tailsteak_fair_sparse_clust, 'mo-', label='TailSTEAK')
# (1) Increased font size for axis labels
ax.set_xlabel("Clustering Coefficient", fontsize=label_fontsize)
ax.set_ylabel("Fairness (1-Gini)", fontsize=label_fontsize)
# (2) Increased font size for axis ticks
ax.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
ax.tick_params(axis='y', labelsize=tick_fontsize)
ax.legend()

# --- Bottom-right: Dense Fairness ---
ax = axes[1, 1]
ax.plot(clust_dense, salsa_fair_dense_clust, 'bo-', label='SALSA')
ax.plot(clust_dense, lgn_fair_dense_clust, 'ro-', label='LightGCN')
ax.plot(clust_dense, pprgo_fair_dense_clust, 'go-', label='PPRGo')
ax.plot(clust_dense, tailsteak_fair_dense_clust, 'mo-', label='TailSTEAK')
# (1) Increased font size for x-axis label
ax.set_xlabel("Clustering Coefficient", fontsize=label_fontsize)
# (2) Increased font size for x-axis ticks
ax.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
ax.legend()

# Adjust layout to prevent title and label overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/plots/clustering_vs_metrics_2x2_edge_swap.pdf")
plt.close()
print("Clustering plots generated.")
# --- END OF MODIFIED SECTION ---


# =============================================================================
# /// PART 2: ANALYSIS FOR DENSITY COMPARISON (6 Custom Pairs) ///
# =============================================================================
print("\n--- Starting Part 2: Density Analysis (6 custom pairs) ---")

# --- Define the 6 custom network pairs ---
pairs_density = [
    ('edge_swap_net2', 'edge_swap_net12'),
    ('edge_swap_net3', 'edge_swap_net13'),
    ('edge_swap_net4', 'edge_swap_net14'),
    ('edge_swap_net6', 'edge_swap_net15'),
    ('edge_swap_net8', 'edge_swap_net16'),
    ('edge_swap_net10', 'edge_swap_net17'),
]
num_pairs_density = len(pairs_density)
sparse_nets_density = [p[0] for p in pairs_density]
dense_nets_density  = [p[1] for p in pairs_density]

# --- Prepare data for Paired Density Plots ---
gnn_properties['density'] = gnn_properties['density'].apply(extract_float)
densities_sparse = gnn_properties.loc[sparse_nets_density, "density"].values
densities_dense  = gnn_properties.loc[dense_nets_density,  "density"].values

# --- Recalculate metrics for the 6 custom pairs ---
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
model_colors = {'SALSA': 'b', 'LightGCN': 'r', 'PPRGo': 'g', 'TailSTEAK': 'o'}

# --- Arrow Plot for Precision ---
plt.figure(figsize=(6, 4))
for name, prec_sparse, prec_dense, _, _ in models_density:
    color = model_colors.get(name, 'k')
    for i in range(num_pairs_density):
        plt.plot([densities_sparse[i], densities_dense[i]], [to_float_safe(prec_sparse[i]), to_float_safe(prec_dense[i])], 'o-', color=color, alpha=0.6)
    plt.plot([], [], 'o-', color=color, label=name)
plt.xlabel("Density")
plt.ylabel("Precision @ 10")
plt.title("Effect of Density Increase (Precision, Custom Pairs)")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/plots/arrow_plot_density_precision_edge_swap_custom.pdf", bbox_inches="tight")
plt.close()

# --- Arrow Plot for Fairness ---
plt.figure(figsize=(6, 4))
for name, _, _, fair_sparse, fair_dense in models_density:
    color = model_colors.get(name, 'k')
    for i in range(num_pairs_density):
        plt.plot([densities_sparse[i], densities_dense[i]], [to_float_safe(fair_sparse[i]), to_float_safe(fair_dense[i])], 'o-', color=color, alpha=0.6)
    plt.plot([], [], 'o-', color=color, label=name)
plt.xlabel("Density")
plt.ylabel("Fairness (1-Gini)")
plt.title("Effect of Density Increase (Fairness, Custom Pairs)")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/plots/arrow_plot_density_fairness_edge_swap_custom.pdf", bbox_inches="tight")
plt.close()

# --- Delta Bar Plot for Precision ---
x = np.arange(num_pairs_density)
bar_width = 0.2
plt.figure(figsize=(12, 6))
for i, (name, prec_sparse, prec_dense, _, _) in enumerate(models_density):
    deltas = np.array([to_float_safe(d) for d in prec_dense]) - np.array([to_float_safe(s) for s in prec_sparse])
    plt.bar(x + i * bar_width, deltas, width=bar_width, label=name, color=model_colors[name])
xtick_labels = [f'N{p[0].split("net")[1]}→N{p[1].split("net")[1]}' for p in pairs_density]
plt.xticks(x + 1.5 * bar_width, xtick_labels)
plt.ylabel('Δ Precision (Dense - Sparse)')
plt.title('Change in Precision (Density Increase, Custom Pairs)')
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/plots/delta_barplot_precision_edge_swap_custom.pdf", bbox_inches="tight")
plt.close()

# --- Delta Bar Plot for Fairness ---
plt.figure(figsize=(12, 6))
for i, (name, _, _, fair_sparse, fair_dense) in enumerate(models_density):
    deltas = np.array([to_float_safe(d) for d in fair_dense]) - np.array([to_float_safe(s) for s in fair_sparse])
    plt.bar(x + i * bar_width, deltas, width=bar_width, label=name, color=model_colors[name])
plt.xticks(x + 1.5 * bar_width, xtick_labels)
plt.ylabel('Δ Fairness (Dense - Sparse)')
plt.title('Change in Fairness (Density Increase, Custom Pairs)')
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/plots/delta_barplot_fairness_edge_swap_custom.pdf", bbox_inches="tight")
plt.close()


# --- Combined Delta Bar Plot for Precision and Fairness ---
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # 2 rows, 1 col, shared x-axis

# --- Top Plot: Delta Precision ---
ax_prec = axes[0]
x = np.arange(num_pairs_density)
bar_width = 0.2

for i, (name, prec_sparse, prec_dense, _, _) in enumerate(models_density):
    deltas = np.array([to_float_safe(d) for d in prec_dense]) - np.array([to_float_safe(s) for s in prec_sparse])
    ax_prec.bar(x + i * bar_width, deltas, width=bar_width, label=name, color=model_colors[name])

# MODIFIED: Increased font size for the y-axis label
ax_prec.set_ylabel('Δ Precision (Dense - Sparse)', fontsize=17)
ax_prec.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax_prec.legend()
# MODIFIED: Increased font size for y-axis ticks for better readability
ax_prec.tick_params(axis='y', labelsize=15)


# --- Bottom Plot: Delta Fairness ---
ax_fair = axes[1]
for i, (name, _, _, fair_sparse, fair_dense) in enumerate(models_density):
    deltas = np.array([to_float_safe(d) for d in fair_dense]) - np.array([to_float_safe(s) for s in fair_sparse])
    ax_fair.bar(x + i * bar_width, deltas, width=bar_width, label=name, color=model_colors[name])

xtick_labels = [f'Net{p[0].split("net")[1]}→Net{p[1].split("net")[1]}' for p in pairs_density]
ax_fair.set_xticks(x + 1.5 * bar_width)

# MODIFIED: Increased font size for the x-axis tick labels
ax_fair.set_xticklabels(xtick_labels, fontsize=16)

# MODIFIED: Increased font size for the y-axis label
ax_fair.set_ylabel('Δ Fairness (Dense - Sparse)', fontsize=17)
ax_fair.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax_fair.legend()
# MODIFIED: Increased font size for y-axis ticks for better readability
ax_fair.tick_params(axis='y', labelsize=15)


# --- Finalize and Save ---
# Added a title for the entire figure for context
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for the title
plt.savefig("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/plots/delta_barplot_combined_edge_swap_custom.pdf", bbox_inches="tight")
plt.close()

print("Combined density delta plot generated.")

print("Density plots for custom pairs generated.")
print("\n✅ All analyses complete.")