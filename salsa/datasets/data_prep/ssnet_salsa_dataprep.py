import os
import pickle
import random
import argparse
from collections import defaultdict

import dgl
import torch
import numpy as np
from tqdm import tqdm
import networkx as nx

class GraphDataPreprocessor:
    """
    Handles the loading, processing, and preparation of graph data for link prediction tasks.

    This class reads training and testing edge lists, generates negative samples
    for evaluation, creates DGL graph objects, and saves all artifacts to disk.
    """
    def __init__(self, network_dir: str):
        """
        Initializes the preprocessor with the source network directory.

        Args:
            network_dir (str): The path to the directory containing 'train.txt' and 'test.txt'.
        """
        self.network_dir = network_dir
        self.rank_dir = os.path.join(network_dir, "eval-rank")
        self.recall_dir = os.path.join(network_dir, "eval-recall")
        
        self.train_edges = None
        self.test_edges = None
        self.adj_set = defaultdict(set)
        self.all_nodes = set()

    def _load_edges(self, file_path: str) -> list[tuple[int, int]]:
        """Loads edges from a text file."""
        edges = []
        with open(file_path, 'r') as f:
            for line in f:
                src, tgt, *_ = line.strip().split()
                edges.append((int(src), int(tgt)))
        return edges

    def _save_pickle(self, data: object, file_path: str):
        """Saves an object to a pickle file and prints a confirmation."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"{os.path.basename(file_path)} saved.")

    def _generate_global_negatives(self, pos_edges: list, num_neg_per_pos: int) -> list:
        """
        Generates unique negative samples from the entire set of nodes.

        This method avoids the infinite loop of the original naive implementation by
        sampling only from the available valid negatives.
        """
        nodes_array = np.array(list(self.all_nodes))
        neg_samples = []

        for u, _ in tqdm(pos_edges, desc="Generating global negative samples"):
            invalid_nodes = self.adj_set[u].copy()
            invalid_nodes.add(u)
            valid_neg = np.setdiff1d(nodes_array, np.array(list(invalid_nodes)), assume_unique=True)
            
            num_to_sample = min(num_neg_per_pos, len(valid_neg))
            
            negs = np.random.choice(valid_neg, size=num_to_sample, replace=False)
            neg_samples.append(negs)
            
        return neg_samples

    def _generate_2hop_negatives(self, pos_edges: list, num_neg_per_pos: int) -> np.ndarray:
        """Generates negative samples from 2-hop neighbors, falling back to global negatives."""
        neg_samples = []
        for u, _ in tqdm(pos_edges, desc="Generating 2-hop negative samples"):
            # Find 2-hop neighbors that are not direct neighbors
            two_hop_neighbors = set()
            for neighbor in self.adj_set[u]:
                two_hop_neighbors.update(self.adj_set[neighbor])
            two_hop_neighbors.discard(u)
            two_hop_neighbors -= self.adj_set[u]

            if len(two_hop_neighbors) >= num_neg_per_pos:
                neg_nodes = random.sample(list(two_hop_neighbors), num_neg_per_pos)
            else:
                # Fallback: use all 2-hop neighbors and fill with global random negatives
                neg_nodes = list(two_hop_neighbors)
                while len(neg_nodes) < num_neg_per_pos:
                    neg = random.choice(list(self.all_nodes))
                    if neg != u and neg not in self.adj_set[u] and neg not in neg_nodes:
                        neg_nodes.append(neg)
            neg_samples.append(neg_nodes)
            
        return np.array(neg_samples)
    
    def _save_np_csr(self, train_edges, out_path):
       
        # 1. Build undirected NetworkX graph
        G = nx.Graph()
        G.add_edges_from(train_edges)
        # 2. Convert to CSR matrix
        adj = nx.to_scipy_sparse_array(G, format='csr')
        # 3. Prepare ptr_array and nei_array as in your function
        ptr_array = np.column_stack((adj.indptr[:-1], np.diff(adj.indptr)))
        nei_array = adj.indices
        out = {'ptr_array': ptr_array, 'nei_array': nei_array}
        # 4. Save
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        print(f"Saved undirected np_csr graph: {os.path.basename(out_path)}")



    def process(self):
        """Runs the full data preparation pipeline."""
        # 1. Create output directories
        os.makedirs(self.rank_dir, exist_ok=True)
        os.makedirs(self.recall_dir, exist_ok=True)
        
        # 2. Load edges and build adjacency info
        print("--- Loading Edges ---")
        self.train_edges = self._load_edges(os.path.join(self.network_dir, 'train.txt'))
        self.test_edges = self._load_edges(os.path.join(self.network_dir, 'test.txt'))
        print(f"Loaded {len(self.train_edges)} training edges and {len(self.test_edges)} test edges.")
        
        for u, v in self.train_edges:
            self.adj_set[u].add(v)
            self.adj_set[v].add(u)
        self.all_nodes = set(self.adj_set.keys())

        # 3. Create and save DGL graphs
        print("\n--- Creating DGL Graphs ---")
        src, dst = map(torch.tensor, zip(*self.train_edges))
        train_graph = dgl.graph((src, dst))
        train_undi_graph = dgl.to_bidirected(train_graph)
        self._save_pickle(train_graph, os.path.join(self.network_dir, "train_graph.pkl"))
        self._save_pickle(train_undi_graph, os.path.join(self.network_dir, "train_undi_graph.pkl"))
        self._save_np_csr(
            self.train_edges,
            os.path.join(self.network_dir, "train_undi_graph.np_csr.pkl")
        )

        # 4. Generate and save negatives for ranking evaluation (2-hop)
        print("\n--- Generating Samples for Ranking (2-hop) ---")
        rank_test_neg = self._generate_2hop_negatives(self.test_edges, num_neg_per_pos=99)
        self._save_pickle(rank_test_neg, os.path.join(self.rank_dir, "rank_test_neg.pkl"))
        self._save_pickle(rank_test_neg, os.path.join(self.rank_dir, "rank_val_neg.pkl")) # Using same for validation
        self._save_pickle(np.array(self.test_edges), os.path.join(self.rank_dir, "rank_test_pos_edgelist.pkl"))
        self._save_pickle(np.array(self.test_edges), os.path.join(self.rank_dir, "rank_val_pos_edgelist.pkl"))

        # 5. Generate and save negatives for recall evaluation (global)
        print("\n--- Generating Samples for Recall (Global) ---")
        recall_test_neg = self._generate_global_negatives(self.test_edges, num_neg_per_pos=999)
        self._save_pickle(recall_test_neg, os.path.join(self.recall_dir, "recall_test_neg.pkl"))
        self._save_pickle(recall_test_neg, os.path.join(self.recall_dir, "recall_val_neg.pkl")) # Using same for validation
        self._save_pickle(np.array(self.test_edges), os.path.join(self.recall_dir, "recall_test_pos_edgelist.pkl"))
        self._save_pickle(np.array(self.test_edges), os.path.join(self.recall_dir, "recall_val_pos_edgelist.pkl"))

        print("\nData preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare graph data for link prediction.")
    parser.add_argument("network_dir", help="Path to the directory containing train.txt and test.txt")
    args = parser.parse_args()
    
    preprocessor = GraphDataPreprocessor(args.network_dir)
    preprocessor.process()