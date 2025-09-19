import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
import argparse
import random
import scipy.io

# Set thread configuration for performance
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)
torch.set_num_interop_threads(16)


def set_seed(seed: int):
    """Sets the seed for all relevant random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_graph_from_mtx(mtx_path):
    # Read the adjacency matrix
    M = scipy.io.mmread(mtx_path).tocoo()
    # Create a NetworkX graph directly from the COO format
    G = nx.Graph()
    # Add edges (skipping self-loops)
    for u, v in zip(M.row, M.col):
        if u != v:
            G.add_edge(u, v)
    return G

def compute_wasserstein_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Wasserstein distance between two degree distributions.
    
    Args:
        pred: Predicted degree distribution
        target: Target degree distribution
        
    Returns:
        Wasserstein distance between the distributions
    """
    pred_sorted, _ = torch.sort(pred)
    target_sorted, _ = torch.sort(target)
    return torch.mean(torch.abs(pred_sorted - target_sorted))

def compute_kl_divergence_degree_distribution(G1: nx.Graph, G2: nx.Graph) -> float:
    """Compute KL divergence between degree distributions of two graphs.
    
    Args:
        G1: First graph
        G2: Second graph
        
    Returns:
        KL divergence between the degree distributions
    """
    deg1 = [deg for _, deg in G1.degree()]
    deg2 = [deg for _, deg in G2.degree()]
    bins = range(0, max(max(deg1), max(deg2)) + 2)
    hist1, _ = np.histogram(deg1, bins=bins, density=True)
    hist2, _ = np.histogram(deg2, bins=bins, density=True)
    return entropy(hist1 + 1e-10, hist2 + 1e-10)

def compute_global_clustering_coefficient(A: torch.Tensor) -> torch.Tensor:
    """Compute global clustering coefficient from adjacency matrix.
    
    Args:
        A: Adjacency matrix
        
    Returns:
        Global clustering coefficient
    """
    A2 = A @ A
    A3 = A2 @ A
    triangles = torch.trace(A3) / 6.0
    deg = A.sum(dim=1)
    triplets = (deg * (deg - 1)).sum() / 2
    triplets = torch.clamp(triplets, min=1e-6)  # prevent NaN division
    return (3 * triangles) / (triplets + 1e-8)


class GCNEncoder(nn.Module):
    """Graph Convolutional Network Encoder."""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNConv(in_channels, hidden_channels))
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        # Last layer
        self.layers.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Node embeddings
        """
        for conv in self.layers[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.layers[-1](x, edge_index)  # no activation on last layer
        return x

class GraphAutoencoder(nn.Module):
    """Graph Autoencoder model."""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, num_layers=num_layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the autoencoder.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Tuple of (reconstructed adjacency scores, node embeddings)
        """
        z = self.encoder(x, edge_index)
        scores = torch.matmul(z, z.T)
        return scores, z


def project_to_graph_with_constraints(
    scores: torch.Tensor,
    triu_idx: torch.Tensor,
    degree_target: torch.Tensor,
    k: int,
    second_pass: bool = True,
    spanning_tree_edges: set = None,
    target_clustering: float = None,
    A_init: torch.Tensor = None,
) -> torch.Tensor:
    """Project scores to graph with degree and clustering constraints.
    
    Args:
        scores: Edge scores
        triu_idx: Upper triangular indices
        degree_target: Target degrees for each node
        k: Number of edges to select
        second_pass: Whether to do a second pass for degree correction
        spanning_tree_edges: Set of edges to enforce (e.g., MST edges)
        clustering_cap: Maximum allowed clustering coefficient
        target_clustering: Target clustering coefficient
        A_init: Initial adjacency matrix
        initial_clustering: Initial clustering coefficient
        
    Returns:
        Binary edge mask satisfying constraints
    """
    N = degree_target.shape[0]
    current_degree = torch.zeros(N, device=scores.device)
    edge_mask = torch.zeros_like(scores)
    A_current = A_init.clone() if A_init is not None else torch.zeros((N, N), device=scores.device)

    # Step 1: Enforce spanning tree edges first
    if spanning_tree_edges is not None:
        for idx in range(scores.shape[0]):
            i = triu_idx[0][idx].item()
            j = triu_idx[1][idx].item()
            if (i, j) in spanning_tree_edges or (j, i) in spanning_tree_edges:
                edge_mask[idx] = 1.0
                current_degree[i] += 1
                current_degree[j] += 1
                A_current[i, j] = 1.0
                A_current[j, i] = 1.0

    # Step 2: Sort candidate edges by score
    sorted_idx = torch.argsort(scores, descending=True)

    # Step 3: Greedily select edges while preserving constraints
    for idx in sorted_idx:
        if edge_mask.sum() >= k:
            break
        i, j = triu_idx[0][idx].item(), triu_idx[1][idx].item()
        if edge_mask[idx] == 1.0:
            continue
        # Degree constraints
        if current_degree[i] >= degree_target[i] or current_degree[j] >= degree_target[j]:
            continue

        # Tentative edge addition
        A_current[i, j] = 1.0
        A_current[j, i] = 1.0
        new_clustering = compute_global_clustering_coefficient(A_current).item()

        if (target_clustering is not None) and (new_clustering > target_clustering):
            A_current[i, j] = 0.0
            A_current[j, i] = 0.0
            continue

        # Accept edge
        edge_mask[idx] = 1.0
        current_degree[i] += 1
        current_degree[j] += 1

    # Step 4: Correct overfilled degrees (optional)
    if second_pass:
        for idx in sorted_idx:
            if edge_mask[idx] == 0:
                continue
            i, j = triu_idx[0][idx].item(), triu_idx[1][idx].item()
            if current_degree[i] > degree_target[i] or current_degree[j] > degree_target[j]:
                edge_mask[idx] = 0.0
                A_current[i, j] = 0.0
                A_current[j, i] = 0.0
                current_degree[i] -= 1
                current_degree[j] -= 1
                if edge_mask.sum() <= k:
                    break

    current_degree = A_current.sum(dim=1).cpu().numpy()
    degree_target_np = degree_target.cpu().numpy()
    l1_error = np.abs(current_degree - degree_target_np).mean()
    l2_error = np.sqrt(((current_degree - degree_target_np)**2).mean())
    clustering_final = compute_global_clustering_coefficient(A_current).item()
    clustering_error = None
    if target_clustering is not None:
        clustering_error = abs(clustering_final - target_clustering)

    print(f"Final Degree L1 Error: {l1_error:.4f}")
    print(f"Final Degree L2 Error: {l2_error:.4f}")
    if clustering_error is not None:
        print(f"Final Clustering Error: {clustering_error:.4f}")

    return edge_mask


def optimize_graph(
    G: nx.Graph,
    hidden_dim: int = 32,
    lr: float = 1e-2,
    epochs: int = 300,
    num_layers: int = 2,
    wass_weight: float = 30.0,
    temperature: float = 0.05,
    alpha: float = 0.05,
    early_stop: bool = False,
    patience: int = 50,
    mst_keep_ratio: float = 1.0,
    target_clustering: float = None,
) -> tuple[nx.Graph, list, list]:
    """Optimize graph structure with degree and clustering constraints.
    
    Args:
        G: Input graph to optimize
        hidden_dim: Hidden dimension of GCN
        lr: Learning rate
        epochs: Number of training epochs
        num_layers: Number of GCN layers
        wass_weight: Weight for Wasserstein loss
        connectivity_weight: Weight for connectivity loss (unused)
        temperature: Temperature for straight-through estimator
        alpha: Degree penalty weight
        early_stop: Whether to use early stopping
        patience: Patience for early stopping
        mst_keep_ratio: Ratio of MST edges to keep
        clustering_cap: Maximum allowed clustering coefficient
        target_clustering: Target clustering coefficient
        clustering_penalty_weight: Weight for clustering penalty
        
    Returns:
        Tuple of (optimized graph, KL history list, final KL list)
    """


    print("[INFO] Starting graph optimization...")
    G = G.copy()
    N = G.number_of_nodes()
    A_orig = nx.to_numpy_array(G)
    degree_orig = A_orig.sum(axis=1)
    degree_orig_torch = torch.tensor(degree_orig, dtype=torch.float32)
    degree_target_int = torch.round(degree_orig_torch).long()
    num_edges = G.number_of_edges()

    # Compute and keep partial MST edges
    T = nx.minimum_spanning_tree(G)
    full_mst_edges = list((min(u, v), max(u, v)) for u, v in T.edges())
    num_keep = int(mst_keep_ratio * len(full_mst_edges))
    partial_spanning_edges = set(full_mst_edges[i] for i in np.random.choice(
        len(full_mst_edges), size=num_keep, replace=False, set_seed=args.seed))
    print(f"Enforcing only {num_keep}/{len(full_mst_edges)} MST edges")

    data = from_networkx(G)
    x = torch.eye(N, requires_grad=False)
    edge_index = data.edge_index
    model = GraphAutoencoder(N, hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_A_sym = None
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in tqdm(range(epochs), desc="Optimizing embeddings"):
        model.train()
        optimizer.zero_grad()

        triu_idx = torch.triu_indices(N, N, offset=1)
        A_scores, _ = model(x, edge_index)
        A_scores_sym = (A_scores + A_scores.T) / 2
        A_scores_sym.fill_diagonal_(0)  # No self-loops
        upper_scores = A_scores_sym[triu_idx[0], triu_idx[1]]

        # Degree-aware bias
        deg_hat = A_scores_sym.sum(dim=1)
        deg_outer = torch.outer(deg_hat, deg_hat)
        deg_target_outer = torch.outer(degree_orig_torch, degree_orig_torch)
        deg_penalty = (deg_outer - deg_target_outer).abs()[triu_idx[0], triu_idx[1]]
        biased_scores = upper_scores - alpha * deg_penalty

        hard_edges = project_to_graph_with_constraints(
            biased_scores,
            triu_idx,
            degree_target_int,
            k=num_edges,
            second_pass=False,
            spanning_tree_edges=partial_spanning_edges,
            target_clustering=target_clustering,
        )

        # Straight-through estimator
        upper_soft = torch.sigmoid(upper_scores / temperature)
        upper_proj = hard_edges.detach() + upper_soft - upper_soft.detach()

        # Reconstruct adjacency
        A_sym = torch.zeros_like(A_scores_sym, dtype=torch.float32)
        A_sym[triu_idx[0], triu_idx[1]] = upper_proj
        A_sym = A_sym + A_sym.T

        # Compute losses
        deg_hat = A_sym.sum(dim=1)        
        wass_loss = compute_wasserstein_distance(deg_hat, degree_orig_torch)

        total_loss = wass_weight * wass_loss

        if target_clustering is not None: 
            clustering_coeff = compute_global_clustering_coefficient(A_sym)

        if torch.isnan(total_loss):
            print(f"[Epoch {epoch:03d}] NaN in loss. Skipping.")
            continue

        total_loss.backward()
        optimizer.step()

        A_np = A_sym.detach().cpu().numpy()
        G_pred = nx.from_numpy_array((A_np > 0.5).astype(float))

        clustering_coeff = compute_global_clustering_coefficient(A_sym)

        density_ok = abs(nx.density(G) - nx.density(G_pred)) < 0.005

        current_wass_loss = wass_loss.item()
        current_clustering = clustering_coeff.item()

        clustering_error = abs(current_clustering - target_clustering) / (target_clustering + 1e-8)
        combined_score = current_wass_loss + clustering_error # we can weight the components

        if 'best_score' not in locals():
            best_score = float('inf')

        if combined_score < best_score and nx.is_connected(G_pred):
            best_score = combined_score
            best_A_sym = A_sym.detach().clone()
            no_improve_epochs = 0
            best_epoch = epoch
            print(f"New best graph at epoch {epoch}: Score={best_score:.4f} (Wass={current_wass_loss:.4f}, Clust={current_clustering:.4f})")
        else:
            no_improve_epochs += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} "
                f"| Clustering: {clustering_coeff.item():.4f} "
                f"| Wasserstein: {current_wass_loss:.4f} | Density OK: {density_ok}"
            )

        if early_stop and no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (last best at {best_epoch})")
            break

    final_matrix_to_project = None
    if best_A_sym is not None:
        print("Using best graph found during training.")
        final_matrix_to_project = best_A_sym

    if final_matrix_to_project is None:
         raise ValueError("Could not generate a graph. No valid matrix was produced.")

    # Use the same constrained projection as in the training loop
    upper_scores = final_matrix_to_project[triu_idx[0], triu_idx[1]]
    
    # Re-calculate biased scores
    deg_hat = final_matrix_to_project.sum(dim=1)
    deg_outer = torch.outer(deg_hat, deg_hat)
    deg_target_outer = torch.outer(degree_orig_torch, degree_orig_torch)
    deg_penalty = (deg_outer - deg_target_outer).abs()[triu_idx[0], triu_idx[1]]
    biased_scores = upper_scores - alpha * deg_penalty

    final_edge_mask = project_to_graph_with_constraints(
        scores=biased_scores,
        triu_idx=triu_idx,
        degree_target=degree_target_int,
        k=num_edges,
        second_pass=False,
        spanning_tree_edges=partial_spanning_edges,
        target_clustering=target_clustering,
    )
    
    A_final = torch.zeros_like(final_matrix_to_project)
    A_final[triu_idx[0], triu_idx[1]] = final_edge_mask
    A_final = A_final + A_final.T

    G_new = nx.from_numpy_array(A_final.cpu().numpy())

    print("Final Clustering:", nx.average_clustering(G_new))
    print("Final KL (actual):", compute_kl_divergence_degree_distribution(G, G_new))
    print("Graph connected:", nx.is_connected(G_new))

    return G_new


def plot_degree_distribution(graphs: list[nx.Graph], labels: list[str], filename: str) -> None:
    """Plot degree distribution comparison between graphs (both log-log and linear-linear).
    
    Args:
        graphs: List of graphs to compare
        labels: Labels for each graph
        filename: Output filename (base, no extension)
    """
    linestyles = ['-', '--', '-.', ':', (0, (5, 1))]
    degrees_hist = []

    # Precompute histograms (for efficiency/reproducibility)
    for G in graphs:
        degrees = [deg for _, deg in G.degree()]
        hist, bins = np.histogram(degrees, bins=range(min(degrees), max(degrees) + 2))
        degrees_hist.append((bins[:-1], hist))

    os.makedirs("plots_selected_10version", exist_ok=True)

    # Plot log-log
    plt.figure(figsize=(8, 6))
    for (bins, hist), label, ls in zip(degrees_hist, labels, linestyles):
        plt.plot(bins, hist, label=label, alpha=0.8, linestyle=ls)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distributions (log-log)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"plots_selected_10version/{filename}_loglog.png", dpi=300)
    plt.close()

    # Plot linear-linear
    plt.figure(figsize=(8, 6))
    for (bins, hist), label, ls in zip(degrees_hist, labels, linestyles):
        plt.plot(bins, hist, label=label, alpha=0.8, linestyle=ls)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distributions (linear)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"plots_selected_10version/{filename}_linear.png", dpi=300)
    plt.close()

def plot_degree_cdf(G1: nx.Graph, G2: nx.Graph, labels: list[str], filename: str) -> None:
    """Plot degree CDF comparison between two graphs.
    
    Args:
        G1: First graph
        G2: Second graph
        labels: Labels for each graph
        filename: Output filename
    """
    plt.figure(figsize=(8, 6))
    for G, label in zip([G1, G2], labels):
        degrees = sorted([d for _, d in G.degree()])
        cdf = np.arange(1, len(degrees) + 1) / len(degrees)
        plt.plot(degrees, cdf, label=label)

    plt.xscale("log")
    plt.xlabel("Degree")
    plt.ylabel("CDF")
    plt.title("Degree Distribution (CDF, log scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    os.makedirs("plots_selected_10version", exist_ok=True)
    plt.savefig(f"plots_selected_10version/{filename}", dpi=300)
    plt.close()

def plot_graph(G: nx.Graph, title: str = "Graph", node_size: int = 20, 
               edge_alpha: float = 0.1, save_path: str = None) -> None:
    """Visualize a graph.
    
    Args:
        G: Graph to visualize
        title: Plot title
        node_size: Size of nodes
        edge_alpha: Transparency of edges
        save_path: Path to save the figure
    """
    pos = nx.spring_layout(G, seed=args.seed)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=0.5)
    
    plt.title(title)
    plt.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wass_weight", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--target_clustering", type=float, default=None) 
    parser.add_argument("--clustering_penalty_weight", type=float, default=5.0) 
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--mst_keep_ratio", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--network1", action="store_true")
    parser.add_argument("--network5", action="store_true")
    parser.add_argument("--out_file_name", type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")


    args = parser.parse_args()
    os.makedirs("saved_graphs_selected_10version", exist_ok=True)

    set_seed(args.seed)

    if args.dense:
        dense_graph_path = "socfb-Reed98_dense/socfb-Reed98.mtx"
        print(f"Loading dense graph from: {dense_graph_path}")
        G = load_graph_from_mtx(dense_graph_path)
        print(f"Dense graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    elif args.sparse:
        sparse_graph_path = "soc-wiki-Vote_sparse/soc-wiki-Vote.mtx"
        print(f"Loading sparse graph from: {sparse_graph_path}")
        G = load_graph_from_mtx(sparse_graph_path)
        print(f"Sparse graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if not nx.is_connected(G):
        print("Graph is not connected")
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    print("Initial clustering:", nx.average_clustering(G))
    print("Initial density:", nx.density(G))
    print("Initial diameter:", nx.diameter(G))

    print("wass_weight: ", args.wass_weight)
    print("alpha: ", args.alpha)
    print("temperature: ", args.temperature)
    print("clustering_cap: ", args.clustering_cap)
    print("target_clustering: ", args.target_clustering)
    print("mst_keep_ratio: ", args.mst_keep_ratio)
    print("dense: ", args.dense)
    print("sparse: ", args.sparse)
    print("num_layers: ", args.num_layers) 
    print("clustering_penalty_weight: ", args.clustering_penalty_weight) 
    print("seed: ", args.seed)


    G_opt = optimize_graph(
                            G,
                            hidden_dim=args.hidden_dim,
                            lr=args.lr,
                            epochs=args.epochs,
                            num_layers=args.num_layers,
                            wass_weight=args.wass_weight,
                            alpha=args.alpha,
                            early_stop=args.early_stop,
                            temperature=args.temperature,
                            mst_keep_ratio=args.mst_keep_ratio,
                            patience=args.patience,
                            target_clustering=args.target_clustering,
                            clustering_penalty_weight=args.clustering_penalty_weight,
                        )
    
    print("Final clustering:", nx.average_clustering(G_opt))
    print("Final density:", nx.density(G_opt))
    print("Density preserved:", abs(nx.density(G) - nx.density(G_opt)) < 0.001)
    print("KL divergence:", compute_kl_divergence_degree_distribution(G, G_opt))
    print("Final diameter:", nx.diameter(G_opt))
    if G_opt.number_of_nodes() <= 5000:  # since avg path length is expensive
        print("Avg. Shortest Path Length:", nx.average_shortest_path_length(G_opt))

    plot_degree_distribution(
        graphs=[G, G_opt],
        labels=["Original", "Rewired"],
        filename=f"degree_comparison_{args.out_file_name}"
    )

    filename_cdf = f"degree_cdf_{args.out_file_name}"
    plot_degree_cdf(G, G_opt, labels=["Original", "Rewired"], filename=filename_cdf)

    plot_graph(G, title="Original Graph", save_path=f"plots_selected_10version/original_subgraph_{args.out_file_name}.png")
    plot_graph(G_opt, title="Rewired Graph", save_path=f"plots_selected_10version/rewired_subgraph_{args.out_file_name}.png")

    os.makedirs("saved_graphs_selected_10version", exist_ok=True)
    nx.write_edgelist(G_opt, f"saved_graphs_selected_10version/rewired_{args.out_file_name}.edgelist")

