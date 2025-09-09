import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import argparse
import io

# ==============================================================================
# --- 1. SHARED UTILITY FUNCTIONS ---
# ==============================================================================

def generalized_base_network(net_name, prefix):
    """
    Extracts the base network name from a versioned name using a dynamic prefix.
    Example: ('gnn_synthetic_network1_v1', 'gnn_synthetic_network') -> 'gnn_synthetic_network1'
    """
    pattern = re.compile(f'({re.escape(prefix)}\\d+)_v\\d+')
    match = pattern.match(str(net_name))
    return match.group(1) if match else net_name

def load_and_tag(path, prefix):
    """Loads a CSV and adds a 'base_net' column for easier grouping."""
    df = pd.read_csv(path)
    df['base_net'] = df['network'].apply(lambda name: generalized_base_network(name, prefix))
    return df

def avg_metric(df, netlist, metric_col, **query):
    """Calculates the average of a given metric for a list of networks."""
    vals = []
    for net in netlist:
        mask = (df['base_net'] == net)
        for k, v in query.items():
            mask &= (df.get(k) == v)
        rows = df[mask]
        vals.append(rows[metric_col].mean() if not rows.empty else np.nan)
    return vals

def extract_float(s):
    """Extracts the first float from a string, e.g., '0.007 (0.000)' -> 0.007."""
    if pd.isnull(s): return np.nan
    return float(str(s).split()[0])

def to_float_safe(val):
    """Safely converts a value to float, returning NaN on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

# ==============================================================================
# --- 2. DATA LOADING & PREPARATION ---
# ==============================================================================

# def load_all_data(args):
#     """Loads all required data files based on command-line arguments."""
#     print("--- Loading data files ---")
#     data = {}
    
#     # --- Load Model Results ---
#     # The prefix is used to correctly group network versions (e.g., gnn_net1_v1, gnn_net1_v2)
#     prefix_map = {
#         'gnn': 'gnn_synthetic_network',
#         'edge_swap': 'edge_swap_net',
#         'ws_pl': '' # WS/PL uses a different matching logic
#     }
#     prefix = prefix_map[args.analysis_type]

#     if prefix: # For GNN and Edge-Swap
#         data['salsa_results'] = load_and_tag(args.salsa_csv, prefix)
#         data['ssnet_results'] = load_and_tag(args.ssnet_csv, prefix)
#         data['tailsteak_results'] = load_and_tag(args.tailsteak_csv, prefix)
#     else: # For WS/PL
#         data['salsa_results'] = pd.read_csv(args.salsa_csv, dtype={'network': str})
#         data['ssnet_results'] = pd.read_csv(args.ssnet_csv, dtype={'network': str})
#         data['tailsteak_results'] = pd.read_csv(args.tailsteak_csv, dtype={'dataset': str})
#         # Standardize columns for WS/PL
#         data['salsa_results'].columns = data['salsa_results'].columns.str.lower().str.replace(' ', '_')
#         data['ssnet_results'].columns = data['ssnet_results'].columns.str.lower()
#         data['tailsteak_results'].columns = data['tailsteak_results'].columns.str.lower().str.replace('@', '')


#     # --- Load Network Properties ---
#     if args.analysis_type in ['gnn', 'edge_swap']:
#         properties = pd.read_csv(args.props_csv).set_index("network")
#         properties['clustering'] = properties['clustering'].apply(extract_float)
#         properties['density'] = properties['density'].apply(extract_float)
#         data['properties'] = properties
#     elif args.analysis_type == 'ws_pl':

#         ws_netdf = pd.read_csv(io.StringIO(ws_properties_data)).set_index('network')
#         pl_netdf = pd.read_csv(io.StringIO(pl_properties_data)).set_index('network')
#         for df in [ws_netdf, pl_netdf]:
#             for col in ['density', 'diameter', 'clustering', 'avg_path', 'fairness']:
#                 df[col] = df[col].apply(extract_float)
#         data['ws_properties'] = ws_netdf
#         data['pl_properties'] = pl_netdf

#     print("✅ Data loading complete.")
#     return data


def load_all_data(args):
    """Loads all required data files based on command-line arguments."""
    print("--- Loading data files ---")
    data = {}
    
    # --- Load Model Results ---
    # The prefix is used to correctly group network versions (e.g., gnn_net1_v1, gnn_net1_v2)
    prefix_map = {
        'gnn': 'gnn_synthetic_network',
        'edge_swap': 'edge_swap_net',
        'ws_pl': '' # WS/PL uses a different matching logic
    }
    prefix = prefix_map[args.analysis_type]

    if prefix: # For GNN and Edge-Swap
        data['salsa_results'] = load_and_tag(args.salsa_csv, prefix)
        data['ssnet_results'] = load_and_tag(args.ssnet_csv, prefix)
        data['tailsteak_results'] = load_and_tag(args.tailsteak_csv, prefix)
    else: # For WS/PL
        data['salsa_results'] = pd.read_csv(args.salsa_csv, dtype={'network': str})
        data['ssnet_results'] = pd.read_csv(args.ssnet_csv, dtype={'network': str})
        data['tailsteak_results'] = pd.read_csv(args.tailsteak_csv, dtype={'dataset': str})
        # Standardize columns for WS/PL
        data['salsa_results'].columns = data['salsa_results'].columns.str.lower().str.replace(' ', '_')
        data['ssnet_results'].columns = data['ssnet_results'].columns.str.lower()
        data['tailsteak_results'].columns = data['tailsteak_results'].columns.str.lower().str.replace('@', '')


    # --- Load Network Properties from the single CSV file ---
    # This now works for all analysis types, including 'ws_pl'.
    print(f"Loading network properties from: {args.props_csv}")
    properties = pd.read_csv(args.props_csv).set_index("network")
    
    # Define columns to parse. Your new CSV has '1-Gini(degree)' instead of 'fairness'.
    # We will rename it for consistency within the script.
    prop_cols_to_parse = ['density', 'diameter', 'clustering', 'avg_path', '1-Gini(degree)']
    for col in prop_cols_to_parse:
        if col in properties.columns:
            properties[col] = properties[col].apply(extract_float)
    
    # Rename for internal consistency
    if '1-Gini(degree)' in properties.columns:
        properties.rename(columns={'1-Gini(degree)': 'fairness'}, inplace=True)

    # For 'ws_pl', we need to split the properties DataFrame
    if args.analysis_type == 'ws_pl':
        # Create boolean masks to identify ws and pl networks
        is_ws = properties.index.str.startswith('ws', na=False)
        is_pl = properties.index.str.startswith('pl', na=False)
        
        data['ws_properties'] = properties[is_ws]
        data['pl_properties'] = properties[is_pl]
    else:
        # For 'gnn' and 'edge_swap', use the properties df directly
        data['properties'] = properties

    print("✅ Data loading complete.")
    return data


# ==============================================================================
# --- 3. ANALYSIS-SPECIFIC LOGIC ---
# ==============================================================================

def run_gnn_analysis(data, output_dir):
    """Performs the analysis and generates plots for GNN-based networks."""
    print("\n--- Running GNN Analysis ---")
    
    # Unpack data
    props = data['properties']
    salsa = data['salsa_results']
    ssnet = data['ssnet_results']
    tailsteak = data['tailsteak_results']
    
    # Define network groups
    sparse_nets = [f'gnn_synthetic_network{i}' for i in range(1, 5)]
    dense_nets = [f'gnn_synthetic_network{i}' for i in range(5, 9)]

    # --- PART 1: Clustering Analysis ---
    clust_sparse = props.loc[sparse_nets, "clustering"].values
    clust_dense = props.loc[dense_nets, "clustering"].values

    salsa_prec_sparse = avg_metric(salsa, sparse_nets, "precision")
    salsa_fair_sparse = avg_metric(salsa, sparse_nets, "1-gini index")
    lgn_prec_sparse = avg_metric(ssnet, sparse_nets, "p10", model="LightGCN", mode="normal")
    lgn_fair_sparse = avg_metric(ssnet, sparse_nets, "1-gini", model="LightGCN", mode="normal")
    pprgo_prec_sparse = avg_metric(ssnet, sparse_nets, "p10", model="PPRGo", mode="normal")
    pprgo_fair_sparse = avg_metric(ssnet, sparse_nets, "1-gini", model="PPRGo", mode="normal")
    tailsteak_prec_sparse = avg_metric(tailsteak, sparse_nets, "p@5")
    tailsteak_fair_sparse = avg_metric(tailsteak, sparse_nets, "1_minus_gini")
    
    salsa_prec_dense = avg_metric(salsa, dense_nets, "precision")
    salsa_fair_dense = avg_metric(salsa, dense_nets, "1-gini index")
    lgn_prec_dense = avg_metric(ssnet, dense_nets, "p10", model="LightGCN", mode="normal")
    lgn_fair_dense = avg_metric(ssnet, dense_nets, "1-gini", model="LightGCN", mode="normal")
    pprgo_prec_dense = avg_metric(ssnet, dense_nets, "p10", model="PPRGo", mode="normal")
    pprgo_fair_dense = avg_metric(ssnet, dense_nets, "1-gini", model="PPRGo", mode="normal")
    tailsteak_prec_dense = avg_metric(tailsteak, dense_nets, "p@5")
    tailsteak_fair_dense = avg_metric(tailsteak, dense_nets, "1_minus_gini")

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
    plt.savefig(os.path.join(output_dir, "gnn_clustering_analysis.pdf"))
    plt.close()
    print("✅ GNN clustering plot saved.")

    # --- PART 2: Density Analysis (Matched Pairs) ---
    pairs_density = [
        ('gnn_synthetic_network1', 'gnn_synthetic_network5'),
        ('gnn_synthetic_network2', 'gnn_synthetic_network6'),
        ('gnn_synthetic_network3', 'gnn_synthetic_network7'),
        ('gnn_synthetic_network4', 'gnn_synthetic_network8'),
    ]
    num_pairs_density = len(pairs_density)
    sparse_nets_density = [p[0] for p in pairs_density]
    dense_nets_density  = [p[1] for p in pairs_density]

    lgn_prec_sparse_density = avg_metric(ssnet, sparse_nets_density, "p10", model="LightGCN", mode="normal")
    lgn_fair_sparse_density = avg_metric(ssnet, sparse_nets_density, "1-gini", model="LightGCN", mode="normal")
    lgn_prec_dense_density  = avg_metric(ssnet, dense_nets_density, "p10", model="LightGCN", mode="normal")
    lgn_fair_dense_density  = avg_metric(ssnet, dense_nets_density, "1-gini", model="LightGCN", mode="normal")

    pprgo_prec_sparse_density = avg_metric(ssnet, sparse_nets_density, "p10", model="PPRGo", mode="normal")
    pprgo_fair_sparse_density = avg_metric(ssnet, sparse_nets_density, "1-gini", model="PPRGo", mode="normal")
    pprgo_prec_dense_density  = avg_metric(ssnet, dense_nets_density, "p10", model="PPRGo", mode="normal")
    pprgo_fair_dense_density  = avg_metric(ssnet, dense_nets_density, "1-gini", model="PPRGo", mode="normal")

    salsa_prec_sparse_density = avg_metric(salsa, sparse_nets_density, "precision")
    salsa_fair_sparse_density = avg_metric(salsa, sparse_nets_density, "1-gini index")
    salsa_prec_dense_density  = avg_metric(salsa, dense_nets_density, "precision")
    salsa_fair_dense_density  = avg_metric(salsa, dense_nets_density, "1-gini index")

    tailsteak_prec_sparse_density = avg_metric(tailsteak, sparse_nets_density, "p@5")
    tailsteak_fair_sparse_density = avg_metric(tailsteak, sparse_nets_density, "1_minus_gini")
    tailsteak_prec_dense_density  = avg_metric(tailsteak, dense_nets_density, "p@5")
    tailsteak_fair_dense_density  = avg_metric(tailsteak, dense_nets_density, "1_minus_gini")

    models_density = [
        ("SALSA", salsa_prec_sparse_density, salsa_prec_dense_density, salsa_fair_sparse_density, salsa_fair_dense_density),
        ("LightGCN", lgn_prec_sparse_density, lgn_prec_dense_density, lgn_fair_sparse_density, lgn_fair_dense_density),
        ("PPRGo", pprgo_prec_sparse_density, pprgo_prec_dense_density, pprgo_fair_sparse_density, pprgo_fair_dense_density),
        ("TailSTEAK", tailsteak_prec_sparse_density, tailsteak_prec_dense_density, tailsteak_fair_sparse_density, tailsteak_fair_dense_density),
    ]
    model_colors = {'SALSA': 'b', 'LightGCN': 'r', 'PPRGo': 'g', 'TailSTEAK': 'm'}

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
    plt.savefig(os.path.join(output_dir, "gnn_density_delta_analysis.pdf"), bbox_inches="tight")
    plt.close()

    print("✅ GNN density plot saved.")

def run_edge_swap_analysis(data, output_dir):
    """Performs the analysis and generates plots for edge-swapped networks."""
    print("\n--- Running Edge-Swap Analysis ---")

    # Unpack data
    props = data['properties']
    salsa = data['salsa_results']
    ssnet = data['ssnet_results']
    tailsteak = data['tailsteak_results']

    # --- PART 1: CLUSTERING ANALYSIS ---
    print("--- Starting Part 1: Clustering Analysis ---")

    # Define network pairs for clustering analysis
    num_pairs_clust = 11
    sparse_nets_clust = [f'edge_swap_net{i}' for i in range(1, num_pairs_clust + 1)]
    dense_nets_clust  = [f'edge_swap_net{i}' for i in range(num_pairs_clust + 1, 2 * num_pairs_clust + 1)]

    # Extract clustering coefficients
    clust_sparse = props.loc[sparse_nets_clust, "clustering"].values
    clust_dense  = props.loc[dense_nets_clust,  "clustering"].values

    # Prepare metrics
    lgn_prec_sparse_clust = avg_metric(ssnet, sparse_nets_clust, "p10", model="LightGCN", mode="normal")
    lgn_fair_sparse_clust = avg_metric(ssnet, sparse_nets_clust, "1-gini", model="LightGCN", mode="normal")
    lgn_prec_dense_clust  = avg_metric(ssnet, dense_nets_clust, "p10", model="LightGCN", mode="normal")
    lgn_fair_dense_clust  = avg_metric(ssnet, dense_nets_clust, "1-gini", model="LightGCN", mode="normal")
    
    pprgo_prec_sparse_clust = avg_metric(ssnet, sparse_nets_clust, "p10", model="PPRGo", mode="normal")
    pprgo_fair_sparse_clust = avg_metric(ssnet, sparse_nets_clust, "1-gini", model="PPRGo", mode="normal")
    pprgo_prec_dense_clust  = avg_metric(ssnet, dense_nets_clust, "p10", model="PPRGo", mode="normal")
    pprgo_fair_dense_clust  = avg_metric(ssnet, dense_nets_clust, "1-gini", model="PPRGo", mode="normal")
    
    salsa_prec_sparse_clust = avg_metric(salsa, sparse_nets_clust, "precision")
    salsa_fair_sparse_clust = avg_metric(salsa, sparse_nets_clust, "1-gini index")
    salsa_prec_dense_clust  = avg_metric(salsa, dense_nets_clust, "precision")
    salsa_fair_dense_clust  = avg_metric(salsa, dense_nets_clust, "1-gini index")
    
    tailsteak_prec_sparse_clust = avg_metric(tailsteak, sparse_nets_clust, "p@5")
    tailsteak_fair_sparse_clust = avg_metric(tailsteak, sparse_nets_clust, "1_minus_gini")
    tailsteak_prec_dense_clust  = avg_metric(tailsteak, dense_nets_clust, "p@5")
    tailsteak_fair_dense_clust  = avg_metric(tailsteak, dense_nets_clust, "1_minus_gini")

    # --- Plotting vs. Clustering ---
    label_fontsize = 17
    tick_fontsize = 13
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='row')

    # Top-left: Sparse Precision
    ax = axes[0, 0]
    ax.plot(clust_sparse, salsa_prec_sparse_clust, 'bo-', label='SALSA')
    ax.plot(clust_sparse, lgn_prec_sparse_clust, 'ro-', label='LightGCN')
    ax.plot(clust_sparse, pprgo_prec_sparse_clust, 'go-', label='PPRGo')
    ax.plot(clust_sparse, tailsteak_prec_sparse_clust, 'mo-', label='TailSTEAK')
    ax.set_ylabel("Precision @ 10", fontsize=label_fontsize)
    ax.set_title("Sparse Networks", fontsize=label_fontsize)
    ax.legend()
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    # Top-right: Dense Precision
    ax = axes[0, 1]
    ax.plot(clust_dense, salsa_prec_dense_clust, 'bo-', label='SALSA')
    ax.plot(clust_dense, lgn_prec_dense_clust, 'ro-', label='LightGCN')
    ax.plot(clust_dense, pprgo_prec_dense_clust, 'go-', label='PPRGo')
    ax.plot(clust_dense, tailsteak_prec_dense_clust, 'mo-', label='TailSTEAK')
    ax.set_title("Dense Networks", fontsize=label_fontsize)
    ax.legend()

    # Bottom-left: Sparse Fairness
    ax = axes[1, 0]
    ax.plot(clust_sparse, salsa_fair_sparse_clust, 'bo-', label='SALSA')
    ax.plot(clust_sparse, lgn_fair_sparse_clust, 'ro-', label='LightGCN')
    ax.plot(clust_sparse, pprgo_fair_sparse_clust, 'go-', label='PPRGo')
    ax.plot(clust_sparse, tailsteak_fair_sparse_clust, 'mo-', label='TailSTEAK')
    ax.set_xlabel("Clustering Coefficient", fontsize=label_fontsize)
    ax.set_ylabel("Fairness (1-Gini)", fontsize=label_fontsize)
    ax.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend()

    # Bottom-right: Dense Fairness
    ax = axes[1, 1]
    ax.plot(clust_dense, salsa_fair_dense_clust, 'bo-', label='SALSA')
    ax.plot(clust_dense, lgn_fair_dense_clust, 'ro-', label='LightGCN')
    ax.plot(clust_dense, pprgo_fair_dense_clust, 'go-', label='PPRGo')
    ax.plot(clust_dense, tailsteak_fair_dense_clust, 'mo-', label='TailSTEAK')
    ax.set_xlabel("Clustering Coefficient", fontsize=label_fontsize)
    ax.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "edge_swap_clustering_analysis.pdf"))
    plt.close()
    print("✅ Edge-swap clustering plot saved.")

    # --- PART 2: DENSITY ANALYSIS ---
    print("\n--- Starting Part 2: Density Analysis (6 custom pairs) ---")
    
    # Define custom network pairs for density analysis
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

    # Recalculate metrics for the custom pairs
    lgn_prec_sparse_density = avg_metric(ssnet, sparse_nets_density, "p10", model="LightGCN", mode="normal")
    lgn_fair_sparse_density = avg_metric(ssnet, sparse_nets_density, "1-gini", model="LightGCN", mode="normal")
    lgn_prec_dense_density  = avg_metric(ssnet, dense_nets_density, "p10", model="LightGCN", mode="normal")
    lgn_fair_dense_density  = avg_metric(ssnet, dense_nets_density, "1-gini", model="LightGCN", mode="normal")
    
    pprgo_prec_sparse_density = avg_metric(ssnet, sparse_nets_density, "p10", model="PPRGo", mode="normal")
    pprgo_fair_sparse_density = avg_metric(ssnet, sparse_nets_density, "1-gini", model="PPRGo", mode="normal")
    pprgo_prec_dense_density  = avg_metric(ssnet, dense_nets_density, "p10", model="PPRGo", mode="normal")
    pprgo_fair_dense_density  = avg_metric(ssnet, dense_nets_density, "1-gini", model="PPRGo", mode="normal")
    
    salsa_prec_sparse_density = avg_metric(salsa, sparse_nets_density, "precision")
    salsa_fair_sparse_density = avg_metric(salsa, sparse_nets_density, "1-gini index")
    salsa_prec_dense_density  = avg_metric(salsa, dense_nets_density, "precision")
    salsa_fair_dense_density  = avg_metric(salsa, dense_nets_density, "1-gini index")
    
    tailsteak_prec_sparse_density = avg_metric(tailsteak, sparse_nets_density, "p@5")
    tailsteak_fair_sparse_density = avg_metric(tailsteak, sparse_nets_density, "1_minus_gini")
    tailsteak_prec_dense_density  = avg_metric(tailsteak, dense_nets_density, "p@5")
    tailsteak_fair_dense_density  = avg_metric(tailsteak, dense_nets_density, "1_minus_gini")

    models_density = [
        ("SALSA", salsa_prec_sparse_density, salsa_prec_dense_density, salsa_fair_sparse_density, salsa_fair_dense_density),
        ("LightGCN", lgn_prec_sparse_density, lgn_prec_dense_density, lgn_fair_sparse_density, lgn_fair_dense_density),
        ("PPRGo", pprgo_prec_sparse_density, pprgo_prec_dense_density, pprgo_fair_sparse_density, pprgo_fair_dense_density),
        ("TailSTEAK", tailsteak_prec_sparse_density, tailsteak_prec_dense_density, tailsteak_fair_sparse_density, tailsteak_fair_dense_density),
    ]
    model_colors = {'SALSA': 'b', 'LightGCN': 'r', 'PPRGo': 'g', 'TailSTEAK': 'm'}

    # --- Combined Delta Bar Plot for Precision and Fairness ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
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
    
    xtick_labels = [f'Net{p[0].split("net")[1]}→Net{p[1].split("net")[1]}' for p in pairs_density]
    ax_fair.set_xticks(x + 1.5 * bar_width)
    ax_fair.set_xticklabels(xtick_labels, fontsize=16)
    ax_fair.set_ylabel('Δ Fairness (Dense - Sparse)', fontsize=17)
    ax_fair.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax_fair.legend()
    ax_fair.tick_params(axis='y', labelsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "edge_swap_density_delta_analysis.pdf"), bbox_inches="tight")
    plt.close()
    print("✅ Edge-swap density plots saved.")

def run_ws_pl_analysis(data, output_dir):
    """Performs the analysis and generates plots for WS and PL networks."""
    print("\n--- Running WS & PL Analysis ---")

    # Unpack data
    ws_netdf = data['ws_properties']
    pl_netdf = data['pl_properties']
    salsa = data['salsa_results']
    ssnet = data['ssnet_results']
    tailsteak = data['tailsteak_results']

    # --- Helper functions moved inside for encapsulation ---
    def get_metric(df, netlist, metric_col, **query):
        """Extracts values of `metric_col` for each network in netlist based on filtering by query."""
        vals = []
        df_net_col = 'network' if 'network' in df.columns else 'dataset'
        
        for net in netlist:
            # Match network names like "pl1" to variations like "pl1_v1", "pl1_v2"
            regex_pattern = f'^{re.escape(net)}(_v\d+)?$'
            conditions = df[df_net_col].str.match(regex_pattern, na=False)
            
            for k, v in query.items():
                conditions &= (df[k] == v)
            
            rows = df[conditions]
            vals.append(rows[metric_col].mean() if not rows.empty else np.nan)
        return vals

    def draw_lines(ax, x_vals, metrics, metric_type):
        """Plots the four models on a given axis for 'prec' or 'fair'."""
        ax.plot(x_vals, metrics[f'salsa_{metric_type}'], 'o-', label='SALSA', color='blue')
        ax.plot(x_vals, metrics[f'lgn_{metric_type}'], 's-', label='LightGCN', color='red')
        ax.plot(x_vals, metrics[f'pprgo_{metric_type}'], '^-', label='PPRGo', color='green')
        ax.plot(x_vals, metrics[f'ts_{metric_type}'], 'd-', label='TailSTEAK', color='m')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    # --- Data gathering logic ---
    ws_clustering_nets = [f'ws{i}' for i in range(1, 21)]
    ws_density_nets    = [f'ws{i}' for i in range(22, 27)]
    pl_clustering_nets = [f'pl{i}' for i in range(1, 21)]
    pl_density_nets    = [f'pl{i}' for i in range(21, 29)]

    ws_x_clust = ws_netdf.loc[ws_clustering_nets, "clustering"].values
    ws_x_dens  = ws_netdf.loc[ws_density_nets, "density"].values
    pl_x_clust = pl_netdf.loc[pl_clustering_nets, "clustering"].values
    pl_x_dens  = pl_netdf.loc[pl_density_nets, "density"].values

    def get_all_metrics(net_list, salsa_df, ssnet_df, tailsteak_df):
        return {
            'salsa_prec': get_metric(salsa_df, net_list, "precision"),
            'salsa_fair': get_metric(salsa_df, net_list, "1-gini_index"),
            'lgn_prec':   get_metric(ssnet_df, net_list, "p10", model="LightGCN", mode="normal"),
            'lgn_fair':   get_metric(ssnet_df, net_list, "1-gini", model="LightGCN", mode="normal"),
            'pprgo_prec': get_metric(ssnet_df, net_list, "p10", model="PPRGo", mode="normal"),
            'pprgo_fair': get_metric(ssnet_df, net_list, "1-gini", model="PPRGo", mode="normal"),
            'ts_prec':    get_metric(tailsteak_df, net_list, "p5"),
            'ts_fair':    get_metric(tailsteak_df, net_list, "1_minus_gini")
        }

    ws_metrics_clust = get_all_metrics(ws_clustering_nets, salsa, ssnet, tailsteak)
    ws_metrics_dens  = get_all_metrics(ws_density_nets, salsa, ssnet, tailsteak)
    pl_metrics_clust = get_all_metrics(pl_clustering_nets, salsa, ssnet, tailsteak)
    pl_metrics_dens  = get_all_metrics(pl_density_nets, salsa, ssnet, tailsteak)
    
    # --- Plotting logic for publication-quality figures ---
    font_axis = {'fontsize': 17}

    # --- FIGURE 1: PRECISION PLOTS ---
    print("--- Generating Figure 1: Precision ---")
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    draw_lines(axes1[0, 0], ws_x_clust, ws_metrics_clust, 'prec')
    axes1[0, 0].set_title('Precision vs. Clustering', fontdict=font_axis)
    axes1[0, 0].set_ylabel('Precision', fontdict=font_axis)
    axes1[0, 0].set_xlabel('Clustering Coefficient', fontdict=font_axis)
    draw_lines(axes1[0, 1], pl_x_clust, pl_metrics_clust, 'prec')
    axes1[0, 1].set_title('Precision vs. Clustering', fontdict=font_axis)
    axes1[0, 1].set_xlabel('Clustering Coefficient', fontdict=font_axis)
    draw_lines(axes1[1, 0], ws_x_dens, ws_metrics_dens, 'prec')
    axes1[1, 0].set_title('Precision vs. Density', fontdict=font_axis)
    axes1[1, 0].set_ylabel('Precision', fontdict=font_axis)
    axes1[1, 0].set_xlabel('Density', fontdict=font_axis)
    draw_lines(axes1[1, 1], pl_x_dens, pl_metrics_dens, 'prec')
    axes1[1, 1].set_title('Precision vs. Density', fontdict=font_axis)
    axes1[1, 1].set_xlabel('Density', fontdict=font_axis)
    fig1.text(0.3, 0.02, '(a) WS-experiment results', ha='center', fontsize=18)
    fig1.text(0.78, 0.02, '(b) PL-experiment results', ha='center', fontsize=18)
    fig1.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.1, hspace=0.3, wspace=0.1)
    plt.savefig(os.path.join(output_dir, "WS_PL_precision_experiments.pdf"), format="pdf")
    plt.close(fig1)

    # --- FIGURE 2: FAIRNESS PLOTS ---
    print("--- Generating Figure 2: Fairness ---")
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    draw_lines(axes2[0, 0], ws_x_clust, ws_metrics_clust, 'fair')
    axes2[0, 0].set_title('Fairness vs. Clustering', fontdict=font_axis)
    axes2[0, 0].set_ylabel('Fairness (1-Gini)', fontdict=font_axis)
    axes2[0, 0].set_xlabel('Clustering Coefficient', fontdict=font_axis)
    draw_lines(axes2[0, 1], pl_x_clust, pl_metrics_clust, 'fair')
    axes2[0, 1].set_title('Fairness vs. Clustering', fontdict=font_axis)
    axes2[0, 1].set_xlabel('Clustering Coefficient', fontdict=font_axis)
    draw_lines(axes2[1, 0], ws_x_dens, ws_metrics_dens, 'fair')
    axes2[1, 0].set_title('Fairness vs. Density', fontdict=font_axis)
    axes2[1, 0].set_ylabel('Fairness (1-Gini)', fontdict=font_axis)
    axes2[1, 0].set_xlabel('Density', fontdict=font_axis)
    draw_lines(axes2[1, 1], pl_x_dens, pl_metrics_dens, 'fair')
    axes2[1, 1].set_title('Fairness vs. Density', fontdict=font_axis)
    axes2[1, 1].set_xlabel('Density', fontdict=font_axis)
    fig2.text(0.3, 0.02, '(a) WS-experiment results', ha='center', fontsize=18)
    fig2.text(0.78, 0.02, '(b) PL-experiment results', ha='center', fontsize=18)
    fig2.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.1, hspace=0.3, wspace=0.1)
    plt.savefig(os.path.join(output_dir, "WS_PL_fairness_experiments.pdf"), format="pdf")
    plt.close(fig2)

    print("✅ WS & PL plots saved.")

# ==============================================================================
# --- 4. MAIN CONTROLLER ---
# ==============================================================================

def setup_arguments():
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run synthetic network analysis and generate plots.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'analysis_type',
        choices=['gnn', 'edge_swap', 'ws_pl'],
        help="The type of analysis to perform:\n"
             "  gnn       - Analyze GNN-generated networks.\n"
             "  edge_swap - Analyze edge-swapped networks.\n"
             "  ws_pl     - Analyze Watts-Strogatz and Power-Law networks."
    )
    parser.add_argument('--salsa-csv', required=True, help="Path to the SALSA results CSV file.")
    parser.add_argument('--ssnet-csv', required=True, help="Path to the SSNet results CSV file.")
    parser.add_argument('--tailsteak-csv', required=True, help="Path to the TailSTEAK results CSV file.")
    parser.add_argument('--props-csv', help="Path to the network properties CSV file (required for 'gnn' and 'edge_swap').")
    parser.add_argument('--output-dir', required=True, help="Directory to save the output plots.")
    
    args = parser.parse_args()
    
    # Validate that props_csv is provided when needed
    if args.analysis_type in ['gnn', 'edge_swap'] and not args.props_csv:
        parser.error("--props-csv is required for 'gnn' and 'edge_swap' analyses.")
        
    return args

def main():
    """Main execution function."""
    args = setup_arguments()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data based on analysis type
    data = load_all_data(args)
    
    # Run the selected analysis
    if args.analysis_type == 'gnn':
        run_gnn_analysis(data, args.output_dir)
    elif args.analysis_type == 'edge_swap':
        run_edge_swap_analysis(data, args.output_dir)
    elif args.analysis_type == 'ws_pl':
        run_ws_pl_analysis(data, args.output_dir)
        
    print("\n✅ Analysis complete.")

if __name__ == '__main__':
    main()