import numpy as np
from tqdm import tqdm
import networkx as nx
import random
from utils.helpers import load_pickle

from scipy.sparse import csr_matrix


# Global structural similarity measures
class GlobalStructuralSimilarity:
    def __init__(self, G_train):
        #self.adj_csr = adj_csr
        self.G_train = G_train
    
    def graph_distance_similarity(self):
        # Compute the shortest path matrix for the graph
        # Using Dijkstra's algorithm for weighted graph, but you can use 'floyd_warshall' for unweighted graphs
        
        #adj_coo = adj_csr.tocoo()
        distance_matrix = dict(nx.all_pairs_shortest_path_length(self.G_train))

        nodes = list(self.G_train.nodes())
        n = len(nodes)
        dist_matrix_np = np.zeros((n, n))

        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                dist_matrix_np[i, j] = distance_matrix[u].get(v, np.inf)  # np.inf if no path exists
        
        # Convert distances to negative for the ranking function
        # f_u(v) = - shortest path distance
        similarity_matrix = -dist_matrix_np # the more the shortest path is the less the probability for linked
        
        return similarity_matrix


# Random Walk-based similarity measures
class RandomWalkSimilarity:

    def __init__(self, train_data_path):
        self.train_network = load_pickle(train_data_path)

    def twitter_wtf_new_version(self, topk_nids_and_ppr_scores_list, node_id, k_for_circle_of_trust=20, tol=1e-8,
                                damping_factor=0.99, k_for_recommendation=-1, salsa_auth=True):
        # Phase 1: Get circle of trust from top-k PPR neighbors
        node_ppr_data = topk_nids_and_ppr_scores_list[node_id]
        top_nodes = node_ppr_data[0]

        if node_id not in top_nodes:
            top_nodes = np.insert(top_nodes, 0, node_id)

        circle_of_trust = top_nodes[:k_for_circle_of_trust]

        # Phase 2: Build bipartite graph with suffixes to distinguish hubs and authorities
        bipartite_graph = nx.DiGraph()
        for node in circle_of_trust:
            bipartite_graph.add_node(f"{node}H")

        for node in circle_of_trust:
            for out_neighbor in self.train_network.successors(node):
                bipartite_graph.add_edge(f"{int(out_neighbor)}A", f"{int(node)}H")

        node_list = list(bipartite_graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        # Build sparse adjacency matrix
        rows, cols = [], []
        for src, dst in bipartite_graph.edges():
            rows.append(node_to_idx[src])
            cols.append(node_to_idx[dst])

        data = np.ones(len(rows), dtype=np.float32)
        A_sparse = csr_matrix((data, (rows, cols)), shape=(len(node_list), len(node_list)))

        # Identify hub and authority node indices
        hub_nodes = [n for n in node_list if n.endswith("H")]
        auth_nodes = [n for n in node_list if n.endswith("A")]
        hub_indices = [node_to_idx[n] for n in hub_nodes]
        auth_indices = [node_to_idx[n] for n in auth_nodes]

        # Build transition matrices
        S_prime = A_sparse[auth_indices, :][:, hub_indices].tocsc()
        R_prime = S_prime.T.tocsc()

        # Normalize columns of sparse matrix
        def normalize_columns_sparse(M):
            col_sums = np.array(M.sum(axis=0)).flatten()
            col_sums[col_sums == 0] = 1.0
            inv_col_sums = 1.0 / col_sums
            return M.multiply(inv_col_sums)

        S_prime = normalize_columns_sparse(S_prime)
        R_prime = normalize_columns_sparse(R_prime)

        # Initialize vectors
        s = np.zeros((len(hub_nodes), 1), dtype=np.float32)
        r = np.zeros((len(auth_nodes), 1), dtype=np.float32)

        target_hub_name = f"{node_id}H"
        if target_hub_name not in hub_nodes:
            return []  # fallback: no central node in graph

        target_hub_idx = hub_nodes.index(target_hub_name)
        s[target_hub_idx] = 1.0

        alpha = 1 - damping_factor
        alpha_vector = np.zeros((len(hub_nodes), 1), dtype=np.float32)
        alpha_vector[target_hub_idx] = alpha

        # Phase 4: SALSA iterations
        tol = float(tol)

        while True:
            r_new = S_prime @ s
            s_new = alpha_vector + (1 - alpha) * (R_prime @ r_new)
            if np.linalg.norm(s_new - s) < tol:
                break
            s = s_new
            r = r_new

        # Phase 5: Ranking
        if salsa_auth:
            relevance_scores = r.flatten()
            node_ids = [int(n[:-1]) for n in auth_nodes]
        else:
            relevance_scores = s.flatten()
            node_ids = [int(n[:-1]) for n in hub_nodes]

        scores = list(zip(node_ids, relevance_scores))
        nonzero = [(n, sc) for n, sc in scores if sc > 0]
        zero = [(n, sc) for n, sc in scores if sc == 0]

        nonzero.sort(key=lambda x: x[1], reverse=True)
        random.shuffle(zero)
        sorted_scores = nonzero + zero

        # Phase 6: Filtering
        neighbors_to_exclude = set(self.train_network.successors(node_id))
        results = []
        for target_node, score in sorted_scores:
            if target_node not in neighbors_to_exclude and target_node != node_id:
                results.append((node_id, target_node, score))
                if 0 < k_for_recommendation == len(results):
                    break

        return results

def predict_links_all(test_edges, adj_csr, k, similarity_func, G_train=None):
    """
    Predict top-k links for each node in the test set using the specified similarity function.
    Supports both directed and undirected models.
    
    Parameters:
    - test_edges: Array of test edges
    - adj_csr: Sparse adjacency matrix in CSR format
    - k: Number of top recommendations
    - similarity_func: Function for undirected similarity
    - G_train: Graph object (optional, required for graph distance similarity)
    """
    
    all_predictions = {}
    

    print(f"Using similarity function: {similarity_func}")
    

    for u in tqdm(set(test_edges[:, 0]), desc="Processing Test Edges"):
        scores = []
        for v in range(len(list(adj_csr.keys()))):
            if u != v:  # Avoid self-comparisons

                if G_train:
                    similarity_matrix = similarity_func()  # Pass G_train if required
                    score = similarity_matrix[u, v]  # Get the similarity score from the precomputed matrix
                    scores.append((v, score))
                else:
                    score = similarity_func(u, v)  # Standard call

                scores.append((v, score))
        
        # Sort by similarity score (highest to lowest)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Store top-k recommendations for node u
        all_predictions[u] = [x[0] for x in sorted_scores[:k]]

    return all_predictions
