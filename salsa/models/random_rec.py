import numpy as np
import random
from tqdm import tqdm

def random_recommendation(test_edges, adj_csr, k, seed=42):
    """
    For each node in test_edges[:,0], recommend k random nodes
    that are not already its neighbors and not itself.
    
    Args:
        test_edges: np.ndarray of shape (n_test_edges, 2), test set edges
        adj_csr: dict or adjacency list-like, adj_csr[u] = neighbors of u
        k: number of recommendations per node
        seed: random seed for reproducibility
    
    Returns:
        predictions: dict {node: [k recommended node ids]}
    """
    np.random.seed(seed)
    random.seed(seed)
    
    nodes = list(adj_csr.keys())
    predictions = {}
    
    users = set(test_edges[:, 0])
    for u in tqdm(users, desc="Random Rec"):
        # Exclude self and existing neighbors
        exclude = set(adj_csr.get(u, []))
        exclude.add(u)
        candidates = [v for v in nodes if v not in exclude]
        
        if len(candidates) >= k:
            recs = random.sample(candidates, k)
        else:
            recs = candidates
        predictions[u] = recs

    return predictions

# Example usage (adapt as needed for integration):
# test_edges = np.array([[0, 2], [1, 3], ...])
# adj_csr = {0: [1, 2], 1: [0, 3], ...}
# k = 10
# preds = random_recommendation(test_edges, adj_csr, k)
