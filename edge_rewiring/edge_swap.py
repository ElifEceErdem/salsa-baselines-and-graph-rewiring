import networkx as nx
import os
import sys
import argparse
import random
import csv


def rewire_to_target_clustering(
    base_graph,
    target_clustering,
    method='random',
    nswap_per_iteration=50,
    max_iterations=5000,
    seed=None,
    tries_multiplier=100
):
    """
    Rewires a network to achieve a target average clustering coefficient.

    This function preserves the degree distribution of the original graph. It can
    use one of two methods:
    1. 'random': Uses the fast `nx.double_edge_swap`. Good for most cases but
       may not reach very low clustering targets.
    2. 'directed': Uses a slower, custom triangle-breaking algorithm that only
       accepts swaps if they reduce the number of triangles. This is more
       effective for achieving very low clustering.

    Args:
        base_graph (nx.Graph): The initial NetworkX graph.
        target_clustering (float): The desired average clustering coefficient.
        method (str): The rewiring strategy to use ('random' or 'directed').
        nswap_per_iteration (int): The number of successful swaps to perform
            in each major iteration.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.
        seed (int): Seed for the random number generator.
        tries_multiplier (int): Multiplier for calculating max swap attempts.

    Returns:
        nx.Graph or None: The rewired graph, or None on failure.
    """
    # --- Input Validation and Setup ---
    if not isinstance(base_graph, nx.Graph) or base_graph.is_directed():
        print("Error: Graph must be an undirected NetworkX Graph.", file=sys.stderr)
        return None

    G = base_graph.copy()
    initial_clustering = nx.average_clustering(G)
    print(f"Initial average clustering: {initial_clustering:.5f}")

    if target_clustering > initial_clustering:
        print(f"Error: Target clustering ({target_clustering:.5f}) is higher than initial ({initial_clustering:.5f}).", file=sys.stderr)
        return None

    print(f"Target average clustering:  {target_clustering:.5f}")
    print(f"Using '{method}' rewiring method.")
    print("-" * 30)

    # --- Dispatch to appropriate rewiring method ---
    if method == 'random':
        return _rewire_randomly(G, target_clustering, nswap_per_iteration, max_iterations, seed, tries_multiplier)
    elif method == 'directed':
        return _rewire_directed(G, target_clustering, nswap_per_iteration, max_iterations, seed, tries_multiplier)
    else:
        print(f"Error: Unknown method '{method}'", file=sys.stderr)
        return None


def _rewire_randomly(G, target_clustering, nswap, max_iter, seed, tries_mult):
    """Helper function for the fast, random double_edge_swap method."""
    current_clustering = nx.average_clustering(G)
    max_tries = nswap * tries_mult
    
    for i in range(max_iter):
        if current_clustering <= target_clustering:
            print(f"\nSuccess! Target reached in {i} iterations.")
            return G
        
        #G = nx.double_edge_swap(G, nswap=nswap, max_tries=max_tries, seed=seed)
        nx.connected_double_edge_swap(G, nswap=nswap, max_tries=max_tries, seed=seed)
        current_clustering = nx.average_clustering(G)
        print(f"Iteration {i+1:>4}: Current Clustering = {current_clustering:.5f}", end='\r')

    print(f"\nWarning: Reached max iterations ({max_iter}). Final clustering is {current_clustering:.5f}")
    return G


def _rewire_directed(G, target_clustering, nswap, max_iter, seed, tries_mult):
    """
    Helper function for the slower, directed triangle-breaking swap method.
    This method is more precise for achieving very low clustering coefficients.
    """
    current_clustering = nx.average_clustering(G)
    rng = random.Random(seed)
    max_tries_per_swap = tries_mult # For this method, the multiplier is per swap

    for i in range(max_iter):
        if current_clustering <= target_clustering:
            print(f"\nSuccess! Target reached in {i} iterations.")
            return G

        swaps_done = 0
        edges = list(G.edges())
        if len(edges) < 2: return G # Cannot perform swaps

        # Try to perform 'nswap' successful swaps in this iteration
        # We give it a generous number of total attempts to find 'nswap' swaps
        for _ in range(nswap * max_tries_per_swap):
            if swaps_done >= nswap:
                break # Reached the desired number of swaps for this iteration

            # 1. Pick two random edges
            try:
                e1, e2 = rng.sample(edges, 2)
            except ValueError:
                break # Not enough edges to sample from
            
            u, v = e1
            x, y = e2

            # 2. Ensure nodes are distinct
            if len({u, v, x, y}) < 4: continue

            # 3. Try swap (u,y) and (v,x)
            if not G.has_edge(u, y) and not G.has_edge(v, x):
                # 4. Calculate triangle change
                tri_before = len(list(nx.common_neighbors(G, u, v))) + len(list(nx.common_neighbors(G, x, y)))
                # Common neighbors for new edges are calculated on the *current* graph
                tri_after = len(list(nx.common_neighbors(G, u, y))) + len(list(nx.common_neighbors(G, v, x)))

                if tri_after < tri_before:
                    G.remove_edges_from([e1, e2])
                    G.add_edges_from([(u, y), (v, x)])
                    swaps_done += 1
                    # Update edge list for next sample
                    edges.remove(e1)
                    edges.remove(e2)
                    edges.extend([(u,y), (v,x)])
        
        if swaps_done == 0:
            print("\nCould not find any more triangle-breaking swaps. The network may be fully randomized.")
            return G
            
        current_clustering = nx.average_clustering(G)
        print(f"Iteration {i+1:>4}: Swaps Done = {swaps_done}, New Clustering = {current_clustering:.5f}", end='\r')

    print(f"\nWarning: Reached max iterations ({max_iter}). Final clustering is {current_clustering:.5f}")
    return G


# helper function to log results to a CSV file
def log_results_to_csv(log_file, data_dict):
    """
    Appends a row of results to a CSV file. Creates the file and header if it doesn't exist.
    """
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        if not file_exists:
            writer.writeheader() # Write header only if file is new
        writer.writerow(data_dict)
    print(f"[INFO] Results logged to '{log_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rewire a network to a target clustering coefficient while preserving the degree distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Input/Output Arguments ---
    parser.add_argument("base_network_file", help="Path to the edgelist file of the base network.")
    parser.add_argument("rewiring_factor", type=float, help="Target clustering as a fraction of the original (e.g., 0.5 for 50%%).")
    parser.add_argument("-o", "--output", default="rewired_network.edgelist", help="Path to save the rewired network.")
    
    # --- Rewiring Control Arguments ---
    parser.add_argument("--method", choices=['random', 'directed'], default='random', help="Rewiring strategy: 'random' is fast, 'directed' is slower but more precise.")
    parser.add_argument("--nswap", type=int, default=100, help="Number of successful swaps per iteration.")
    parser.add_argument("--tries-multiplier", type=int, default=100, help="Multiplier for calculating max swap attempts in an iteration.")
    
    # --- Reproducibility and Connectivity Arguments ---
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--max-connect-tries", type=int, default=10000, help="Maximum number of attempts to get a connected graph.") # NEW
    
    # --- Logging Argument ---
    parser.add_argument("--log-file", default="rewiring_log.csv", help="Path to the CSV file for logging results.") # NEW
    
    args = parser.parse_args()

    if not os.path.exists(args.base_network_file):
        print(f"Error: Base network file not found at '{args.base_network_file}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading base network from '{args.base_network_file}'...")
    base_graph = nx.read_edgelist(args.base_network_file, nodetype=int)
    
    # It's important to start with a connected graph for this process to be meaningful.
    if not nx.is_connected(base_graph):
        print("Warning: The base network is not connected. The rewiring will proceed on the largest connected component.", file=sys.stderr)
        largest_cc = max(nx.connected_components(base_graph), key=len)
        base_graph = base_graph.subgraph(largest_cc).copy()

    print(f"Using network component with {base_graph.number_of_nodes()} nodes and {base_graph.number_of_edges()} edges.")
    print("-" * 30)

    initial_clustering = nx.average_clustering(base_graph)
    target_clustering = initial_clustering * args.rewiring_factor

    rewired_graph = None
    for i in range(args.max_connect_tries):
        print(f"\n--- Attempt {i+1}/{args.max_connect_tries} to generate a connected rewired graph ---")
        # Use a different seed for each attempt to get different results, unless no seed is given.
        current_seed = args.seed + i if args.seed is not None else None
        
        temp_graph = rewire_to_target_clustering(
            base_graph=base_graph,
            target_clustering=target_clustering,
            method=args.method,
            seed=current_seed,
            nswap_per_iteration=args.nswap,
            tries_multiplier=args.tries_multiplier
        )

        if temp_graph and nx.is_connected(temp_graph):
            print(f"\n[SUCCESS] Attempt {i+1} resulted in a connected graph.")
            rewired_graph = temp_graph
            break # Exit the loop on success
        elif temp_graph:
            print(f"\n[INFO] Attempt {i+1} resulted in a disconnected graph. Retrying...")
        else:
            print(f"\n[ERROR] Rewiring failed on attempt {i+1}. Retrying...")
    
    # --- Check if the loop failed ---
    if rewired_graph is None:
        print(f"\n[FATAL] Failed to generate a connected graph after {args.max_connect_tries} attempts.", file=sys.stderr)
        sys.exit(1)

    # --- Proceed with the final, connected graph ---
    final_clustering = nx.average_clustering(rewired_graph)
    print("\n" + "=" * 30)
    print("--- Final Results ---")
    print(f"Initial Clustering: {initial_clustering:.5f}")
    print(f"Target Clustering:  ~{target_clustering:.5f}")
    print(f"Final Clustering:   {final_clustering:.5f}")

    degrees1 = sorted([d for n, d in base_graph.degree()])
    degrees2 = sorted([d for n, d in rewired_graph.degree()])
    if degrees1 == degrees2:
        print("\n[SUCCESS] Degree distribution was preserved.")
    else:
        print("\n[ERROR] Degree distribution was altered.", file=sys.stderr)

    nx.write_edgelist(rewired_graph, args.output, data=False)
    print(f"[INFO] Rewired network saved to '{args.output}'")

    log_data = {
        'output_file': args.output,
        'initial_clustering': f"{initial_clustering:.5f}",
        'target_clustering': f"{target_clustering:.5f}",
        'final_clustering': f"{final_clustering:.5f}"
    }
    log_results_to_csv(args.log_file, log_data)