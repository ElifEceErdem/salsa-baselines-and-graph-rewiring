import os
import random
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_degree_histogram(graph, subset_edges, name, filename):
    plt.figure(figsize=(7, 5))
    full_degrees = [d for _, d in graph.degree()]
    train_graph = nx.Graph()
    train_graph.add_nodes_from(graph.nodes())
    train_graph.add_edges_from(subset_edges)
    train_degrees = [d for _, d in train_graph.degree()]
    plt.hist(full_degrees, bins=50, alpha=0.5, label="Original", density=True)
    plt.hist(train_degrees, bins=50, alpha=0.5, label="Train", density=True)
    plt.title("Degree Distribution (Original vs Train)")
    plt.xlabel("Degree")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def split_edges(graph, val_size=25000, test_size=50000, plot_file="train_vs_original.png"):
    all_edges = list(graph.edges())
    degrees = dict(graph.degree())
    degree_bins = defaultdict(list)
    for u, v in all_edges:
        bin_key = (degrees[u] + degrees[v]) // 2
        degree_bins[bin_key].append((u, v))
    val_edges, test_edges, train_edges = [], [], []
    val_target, test_target = val_size, test_size
    for edges in degree_bins.values():
        random.shuffle(edges)
        num_val = min(val_target, len(edges) // 3)
        num_test = min(test_target, len(edges) // 3)
        val_edges.extend(edges[:num_val])
        test_edges.extend(edges[num_val:num_val + num_test])
        train_edges.extend(edges[num_val + num_test:])
        val_target -= num_val
        test_target -= num_test
    print("# Final splits:", len(train_edges), len(val_edges), len(test_edges))
    plot_degree_histogram(graph, train_edges, "train", plot_file)
    return train_edges, val_edges, test_edges


def safe_edge_split(G, val_size=25000, test_size=50000):
    all_edges = list(G.edges())
    random.shuffle(all_edges)
    train_edges, val_edges, test_edges = [], [], []

    degree = dict(G.degree())
    
    train_degree = degree.copy()

    total_val, total_test = 0, 0
    for u, v in all_edges:
        # Check if this is the last edge for either node in the train set
        if train_degree[u] <= 1 or train_degree[v] <= 1:
            train_edges.append((u, v))
            train_degree[u] -= 1
            train_degree[v] -= 1
        elif total_val < val_size:
            val_edges.append((u, v))
            train_degree[u] -= 1
            train_degree[v] -= 1
            total_val += 1
        elif total_test < test_size:
            test_edges.append((u, v))
            train_degree[u] -= 1
            train_degree[v] -= 1
            total_test += 1
        else:
            train_edges.append((u, v))
            train_degree[u] -= 1
            train_degree[v] -= 1
    print("# Final splits:", len(train_edges), len(val_edges), len(test_edges))
    return train_edges, val_edges, test_edges


def save_edge_list(edges, filename):
    with open(filename, "w") as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\t1\n")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified edge split for edgelist graph.")
    parser.add_argument("input_edgelist", help="Input .edgelist file path")
    parser.add_argument("--val_size", type=int, default=25000, help="Validation set size")
    parser.add_argument("--test_size", type=int, default=50000, help="Test set size")
    args = parser.parse_args()

    input_edgelist = args.input_edgelist
    out_dir = os.path.dirname(input_edgelist)
    basename = os.path.splitext(os.path.basename(input_edgelist))[0]
    plot_file = os.path.join(out_dir, f"{basename}_train_vs_original.png")

    # Load graph
    G = nx.read_edgelist(input_edgelist, delimiter=" ", nodetype=int, data=True)

    # Split edges
    # train_edges, val_edges, test_edges = split_edges(G, val_size=args.val_size, test_size=args.test_size,
    #                                                 plot_file=plot_file)
    
    train_edges, val_edges, test_edges = safe_edge_split(G, val_size=args.val_size, test_size=args.test_size)


    # Save files
    save_edge_list(train_edges, os.path.join(out_dir, "train.txt"))
    save_edge_list(val_edges, os.path.join(out_dir, "val.txt"))
    save_edge_list(test_edges, os.path.join(out_dir, "test.txt"))

    print("Edge splits saved to:", out_dir)
