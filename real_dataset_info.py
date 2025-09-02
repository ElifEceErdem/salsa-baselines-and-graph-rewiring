import os
import networkx as nx
import numpy as np
import pandas as pd

# --- Configuration ---
# Define the datasets to be analyzed with their respective paths
# Using a dictionary helps label the output clearly.
FILE_PATHS = {
    "Twitter": "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/a_repo/salsa/datasets/twitter/twitter.edgelist",
    "Wiki-Vote": "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/a_repo/salsa/datasets/wiki/soc-wiki-Vote.edgelist",
    "Facebook-Reed98": "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/a_repo/salsa/datasets/reed98/socfb-Reed98.edgelist",
    "Pokec": "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/a_repo/salsa/datasets/pokec/soc-pokec-relationships_1.txt",

}

# --- Helper Functions ---
def calculate_gini(values):
    """Calculates the Gini coefficient for a list of values."""
    values = np.array(values, dtype=float).flatten()
    if np.amin(values) < 0:
        values -= np.amin(values) # shift values to be non-negative
    if np.sum(values) == 0:
        return 0.0
    values = np.sort(values)
    n = values.shape[0]
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
    return gini

# --- Main Metric Computation Function ---
def compute_network_metrics(G, dataset_name):
    """
    Computes a set of metrics for a given graph.
    Handles disconnected graphs by calculating path-based metrics on the Largest Connected Component (LCC).
    """
    print(f"🔬 Analyzing '{dataset_name}'...")
    
    # --- Basic Metrics (calculated on the whole graph) ---
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes == 0:
        # Return empty metrics for an empty graph
        return {
            '# Nodes': 0, '# Edges': 0, 'Density': 0, 'Avg. Degree': 0,
            'Avg. Clustering': 0, 'Avg. Path (LCC)': np.nan, '1-Gini(Degree)': np.nan
        }

    density = nx.density(G)
    avg_degree = 2 * num_edges / num_nodes
    
    # --- Degree Gini (calculated on the whole graph) ---
    degree_values = [d for n, d in G.degree()]
    gini_degree = calculate_gini(degree_values)
    one_minus_gini = 1 - gini_degree

    print(f"  -> {num_nodes:,} nodes, {num_edges:,} edges, Density: {density:.4e}, Avg. Degree: {avg_degree:.2f}, 1-Gini(Degree): {one_minus_gini:.4f}")

    # --- Metrics that can be slow or require graph connectivity ---
    print(f"  -> Calculating average clustering...")
    #avg_clustering = nx.average_clustering(G)
    
    # For path-based metrics, we must handle disconnected graphs
    # if nx.is_connected(G):
    #     print(f"  -> Graph is connected. Calculating average shortest path...")
    #     avg_path = nx.average_shortest_path_length(G)
    # else:
    #     print(f"  -> ⚠️ Graph is disconnected. Using the Largest Connected Component (LCC) for path metrics.")
    #     # Find the largest connected component
    #     largest_cc = max(nx.connected_components(G), key=len)
    #     G_lcc = G.subgraph(largest_cc)
    #     nodes_in_lcc = G_lcc.number_of_nodes()
    #     print(f"  -> LCC has {nodes_in_lcc:,} nodes ({nodes_in_lcc/num_nodes:.2%}) of the total.")
    #     print(f"  -> Calculating average shortest path on LCC (this may take a while)...")
    #     avg_path = nx.average_shortest_path_length(G_lcc)

    return {
        '# Nodes': num_nodes,
        '# Edges': num_edges,
        'Density': density,
        'Avg. Degree': avg_degree,
        # 'Avg. Clustering': avg_clustering,
        # 'Avg. Path (LCC)': avg_path,
        '1-Gini(Degree)': one_minus_gini
    }

# --- Main Processing ---
results = []
print("--- Starting Network Analysis ---")

for name, path in FILE_PATHS.items():
    print("-" * 50)
    if not os.path.exists(path):
        print(f"❌ File not found, skipping: {path}")
        continue
    
    try:
        # Load the graph, handling specific delimiters or comment characters
        if 'pokec' in name.lower():
            G = nx.read_edgelist(path, nodetype=int, comments='#', delimiter='\t')
        else:
            G = nx.read_edgelist(path, nodetype=int, comments='#')
            
        print(f"📂 Loaded graph '{name}' with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        # Convert to a simple, undirected graph for consistent metric calculation
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))

        # Compute all metrics for the loaded graph
        metrics = compute_network_metrics(G, name)
        metrics['Dataset'] = name  # Add dataset name for indexing
        results.append(metrics)
        
    except Exception as e:
        print(f"❌ An error occurred while processing {path}: {e}")

# --- Display Results ---
if results:
    # Create and format the DataFrame for display
    df = pd.DataFrame(results).set_index('Dataset')
    
    # Reorder columns for a logical presentation
    column_order = [
        '# Nodes', '# Edges', 'Density', 'Avg. Degree', 
        'Avg. Clustering', 'Avg. Path (LCC)', '1-Gini(Degree)'
    ]
    df = df[column_order]

    # --- Console Output ---
    df_display = df.copy()
    df_display['# Nodes'] = df_display['# Nodes'].map('{:,.0f}'.format)
    df_display['# Edges'] = df_display['# Edges'].map('{:,.0f}'.format)
    df_display['Density'] = df_display['Density'].map('{:.2e}'.format) # Use scientific notation
    
    print("\n" + "=" * 90)
    print("--- 📊 Network Metrics Summary ---")
    print("Note: 'Avg. Path (LCC)' is calculated on the Largest Connected Component for disconnected graphs.")
    print("=" * 90)
    print(df_display.to_string(float_format="%.4f"))

    # --- LaTeX Output ---
    df_latex = df.copy()
    # Rename columns for a cleaner LaTeX table
    df_latex.rename(columns={'Avg. Clustering': 'Clustering', 'Avg. Path (LCC)': 'Avg. Path'}, inplace=True)
    df_latex['# Nodes'] = df_latex['# Nodes'].map('{:,.0f}'.format)
    df_latex['# Edges'] = df_latex['# Edges'].map('{:,.0f}'.format)
    df_latex['Density'] = df_latex['Density'].apply(lambda x: f"${x:.2e}$") # Format as math
    
    print("\n" + "=" * 90)
    print("--- 📋 LaTeX Output ---")
    print("=" * 90)
    print(df_latex.to_latex(float_format="%.3f", escape=False, column_format='lrrrrrr'))

else:
    print("\nNo data was processed. Please verify the file paths.")