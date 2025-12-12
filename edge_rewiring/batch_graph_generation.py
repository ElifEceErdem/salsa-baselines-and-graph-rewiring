import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from datetime import datetime
import csv

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix, csr_matrix

# -------------------------------
# Utilities
# -------------------------------

def degree_sequence(G: nx.Graph) -> Dict[int, int]:
    return {u: G.degree(u) for u in G.nodes()}


def triangles_count(G: nx.Graph) -> int:
    tri_by_node = nx.triangles(G)
    return int(sum(tri_by_node.values()) // 3)


def num_triplets_from_degrees(deg: Dict[int, int]) -> int:
    return int(sum(d * (d - 1) // 2 for d in deg.values()))


def common_neighbors_fast(N, u, v) -> int:
    if len(N[u]) > len(N[v]):
        u, v = v, u
    return sum((w in N[v]) for w in N[u])


def _largest_cc(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or nx.is_connected(G):
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def graph_diameter(G: nx.Graph) -> int:
    H = _largest_cc(G)
    if H.number_of_nodes() <= 1:
        return 0
    try:
        from networkx.algorithms import approximation as nx_app
        return nx_app.diameter(H)
    except Exception:
        return nx.diameter(H)


def effective_diameter(G: nx.Graph, q: float = 0.9, k: int = 200, seed: int = 42) -> float:
    rng = random.Random(seed)
    H = _largest_cc(G)
    n = H.number_of_nodes()
    if n <= 1:
        return 0.0
    nodes = list(H.nodes())
    sources = [rng.choice(nodes) for _ in range(min(k, n))]
    dists = []
    for s in sources:
        lengths = nx.single_source_shortest_path_length(H, s)
        dists.extend([d for _, d in lengths.items() if d > 0])
    return float(np.quantile(dists, q)) if dists else 0.0


def load_graph_from_mtx(path: str) -> nx.Graph:
    A = mmread(path)
    if not isinstance(A, (coo_matrix, csr_matrix)):
        A = coo_matrix(A)
    A = A.tocsr()
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G

# -------------------------------
# Encoder & scoring (centered cosine on SGC with A^2 skip)
# -------------------------------

def simple_sgc_encoder(
    G: nx.Graph,
    K: int = 10,
    out_dim: int = 64,
    seed: int = 42,
    twohop_eta: float = 0.6,
    a2_row_cap: Optional[float] = 0.05,
    add_self_loops: bool = True,
    normalize_rows: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
    A = A.tocsr()
    A.setdiag(0); A.eliminate_zeros()

    A_sl = (A + sp.eye(n, format="csr")) if add_self_loops else A
    deg = np.array(A_sl.sum(1)).ravel()
    deg_safe = np.maximum(deg, 1e-12)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_safe))
    Ahat = D_inv_sqrt @ A_sl @ D_inv_sqrt

    if twohop_eta > 0.0:
        A2 = (A @ A).tocsr()
        A2.setdiag(0); A2.eliminate_zeros()
        rowsum = np.maximum(np.asarray(A2.sum(1)).ravel(), 1e-12)
        B = A2.multiply(1.0 / rowsum[:, None])
        if a2_row_cap is not None:
            Bcoo = B.tocoo(copy=True)
            data = np.minimum(Bcoo.data, float(a2_row_cap))
            B = sp.csr_matrix((data, (Bcoo.row, Bcoo.col)), shape=B.shape)
        F = 0.5 * (Ahat + twohop_eta * B + (Ahat + twohop_eta * B).T)
    else:
        F = Ahat

    X = np.eye(n, dtype=np.float32)
    H = X
    for _ in range(K):
        H = F @ H

    W = rng.standard_normal((H.shape[1], out_dim)) / math.sqrt(H.shape[1])
    Z = H @ W
    if normalize_rows:
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return Z


def make_centered_cosine_scorer(Z: np.ndarray, degrees: Dict[int, int], deg_power: float = 1.0):
    Z = np.asarray(Z, dtype=np.float32)
    n, _ = Z.shape
    deg = np.array([degrees[i] for i in range(n)], dtype=np.float32)
    w = np.power(np.maximum(deg, 0.0), deg_power)
    w = w / (w.sum() + 1e-12)
    m = (w[:, None] * Z).sum(axis=0, keepdims=True)
    Zc = Z - m
    norms = np.linalg.norm(Zc, axis=1) + 1e-12

    def score(u, v):
        a = Zc[u]; b = Zc[v]
        return float((a @ b) / (norms[u] * norms[v]))

    return score

# -------------------------------
# Spanning tree (freeze connectivity)
# -------------------------------

def spanning_tree_from_base_graph(G_base: nx.Graph, score_fn) -> Tuple[Set[Tuple[int, int]], Dict[int, int]]:
    H = nx.Graph()
    H.add_nodes_from(G_base.nodes())
    for u, v in G_base.edges():
        H.add_edge(u, v, weight=score_fn(u, v))
    T = nx.maximum_spanning_tree(H, weight="weight")
    frozen = set(tuple(sorted(e)) for e in T.edges())
    deg_T = defaultdict(int)
    for u, v in frozen:
        deg_T[u] += 1
        deg_T[v] += 1
    return frozen, deg_T

# -------------------------------
# Candidate set (full) & projection
# -------------------------------

def build_full_candidate_set(n: int, frozen_edges: Set[Tuple[int, int]], score_fn):
    cand = {u: [] for u in range(n)}
    weight = {}
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in frozen_edges:
                continue
            cand[u].append(v)
            cand[v].append(u)
            weight[(u, v)] = score_fn(u, v)
    for u in range(n):
        cand[u].sort()
    return cand, weight


def weighted_havel_hakimi(n, residual_deg, cand, weight, forbidden_edges=frozenset()):
    rem = residual_deg.copy()
    S = set()
    avail = {u: set() for u in range(n)}
    for a, nbrs in cand.items():
        for b in nbrs:
            if (a, b) not in forbidden_edges and (b, a) not in forbidden_edges:
                avail[a].add(b)
                avail.setdefault(b, set()).add(a)

    def eff_w(u, v):
        return weight.get((min(u, v), max(u, v)), 0.0)

    while True:
        u = max(rem, key=lambda x: rem[x])
        if rem[u] == 0:
            if all(d == 0 for d in rem.values()):
                break
            u = max([x for x in rem if rem[x] > 0], key=lambda x: rem[x])
        r = rem[u]
        feas = []
        for v in avail.get(u, ()):
            if rem.get(v, 0) <= 0:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in S or (a, b) in forbidden_edges:
                continue
            feas.append((eff_w(u, v), v))
        if len(feas) < r:
            raise ValueError("Residual degree sequence not realizable on candidate set.")
        feas.sort(reverse=True)
        chosen_vs = [v for _, v in feas[:r]]
        for v in chosen_vs:
            a, b = (u, v) if u < v else (v, u)
            S.add((a, b))
            rem[u] -= 1
            rem[v] -= 1
    return S


def project_bmatching_min_cost_flow(n, residual_deg, cand, weight, frozen_edges):
    F = nx.DiGraph()
    src, sink = "_src", "_sink"
    F.add_node(src); F.add_node(sink)
    L = {u: f"L{u}" for u in range(n)}
    R = {u: f"R{u}" for u in range(n)}
    total = sum(residual_deg.values())
    if total % 2 != 0:
        raise ValueError("Residual degree sum must be even.")
    for u in range(n):
        d = residual_deg[u]
        if d > 0:
            F.add_edge(src, L[u], capacity=d, weight=0)
            F.add_edge(R[u], sink, capacity=d, weight=0)
    scale = 1e6
    seen = set()
    for a, nbrs in cand.items():
        for b in nbrs:
            u, v = (a, b) if a < b else (b, a)
            if u == v or (u, v) in seen or (u, v) in frozen_edges:
                continue
            seen.add((u, v))
            w = weight.get((u, v), 0.0)
            F.add_edge(L[u], R[v], capacity=1, weight=int(round(-scale * w)))
    flow = nx.algorithms.flow.min_cost_flow(F)
    chosen = set(); used = defaultdict(int)
    for u in range(n):
        Lu = L[u]
        for rv, f in flow.get(Lu, {}).items():
            if f <= 0 or not rv.startswith("R"):
                continue
            v = int(rv[1:])
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in frozen_edges:
                continue
            chosen.add((a, b)); used[u] += 1; used[v] += 1
    for u in range(n):
        if used.get(u, 0) != residual_deg[u]:
            raise ValueError(f"Flow projection did not realize residual degrees for node {u}")
    return chosen

# -------------------------------
# Degree-preserving annealing
# -------------------------------

def anneal_to_target_clustering(G: nx.Graph, target_C: float, frozen_edges=frozenset(), steps: int = 100_000, T0: float = 1.5, Tf: float = 0.005, rng=None) -> nx.Graph:
    if rng is None:
        rng = random.Random(0)
    n = G.number_of_nodes()
    edges = set(tuple(sorted(e)) for e in G.edges())
    nonfrozen = [e for e in edges if e not in frozen_edges]
    N = {u: set(G.neighbors(u)) for u in G.nodes()}
    T = triangles_count(G)
    triplets = num_triplets_from_degrees(degree_sequence(G))
    target_T = target_C * triplets / 3.0

    def edge_triangle(u, v):
        return common_neighbors_fast(N, u, v)

    def accept(oldS, newS, temp):
        if newS < oldS:
            return True
        return rng.random() < math.exp(-(newS - oldS) / max(temp, 1e-9))

    def propose_swap():
        (a, b) = rng.choice(nonfrozen)
        (c, d) = rng.choice(nonfrozen)
        if len({a, b, c, d}) < 4:
            return None
        cand1 = (min(a, c), max(a, c)), (min(b, d), max(b, d))
        cand2 = (min(a, d), max(a, d)), (min(b, c), max(b, c))
        return (a, b), (c, d), cand1, cand2

    S = abs(T - target_T)
    for t in range(steps):
        temp = T0 * (Tf / T0) ** (t / max(steps - 1, 1))
        p = propose_swap()
        if p is None:
            continue
        (a, b), (c, d), (e1, e2), (f1, f2) = p
        for x, y in ((e1, e2), (f1, f2)):
            if x in edges or y in edges or x in frozen_edges or y in frozen_edges:
                continue
            u1, v1 = x; u2, v2 = y
            delta = (edge_triangle(*x) + edge_triangle(*y) - edge_triangle(a, b) - edge_triangle(c, d))
            newT = T + delta
            newS = abs(newT - target_T)
            if accept(S, newS, temp):
                edges.remove((a, b)); edges.remove((c, d))
                nonfrozen.remove((a, b)); nonfrozen.remove((c, d))
                N[a].remove(b); N[b].remove(a); N[c].remove(d); N[d].remove(c)
                edges.add(x); edges.add(y)
                nonfrozen.append(x); nonfrozen.append(y)
                N[u1].add(v1); N[v1].add(u1); N[u2].add(v2); N[v2].add(u2)
                T = newT; S = newS
                break

    H = nx.Graph(); H.add_nodes_from(range(n)); H.add_edges_from(edges)
    return H

# -------------------------------
# Orchestrator
# -------------------------------

def rewire_with_hh_and_annealing(
    G_base: nx.Graph,
    target_clustering: float,
    anneal_steps: int = 100_000,
    K: int = 10,
    twohop_eta: float = 0.6,
    a2_row_cap: Optional[float] = 0.05,
    deg_power: float = 1.0,
    seed: int = 42,
    T0: float = 1.5,
    Tf: float = 0.005,
):
    rng = random.Random(seed)
    n = G_base.number_of_nodes()

    Z = simple_sgc_encoder(
        G_base, K=K, out_dim=64, seed=seed,
        twohop_eta=twohop_eta, a2_row_cap=a2_row_cap,
        add_self_loops=True, normalize_rows=False,
    )
    score_fn = make_centered_cosine_scorer(Z, degrees=dict(G_base.degree()), deg_power=deg_power)

    frozen_edges, deg_T = spanning_tree_from_base_graph(G_base, score_fn)
    nodes = list(range(n))
    deg_target = degree_sequence(G_base)
    residual_deg = {u: deg_target[u] - deg_T.get(u, 0) for u in nodes}
    if any(d < 0 for d in residual_deg.values()):
        raise ValueError("Frozen edges exceed target degrees for some nodes.")

    cand, weight = build_full_candidate_set(n, frozen_edges, score_fn)

    try:
        chosen = weighted_havel_hakimi(n, residual_deg, cand, weight, forbidden_edges=frozen_edges)
    except ValueError:
        chosen = project_bmatching_min_cost_flow(n, residual_deg, cand, weight, frozen_edges)

    G_init = nx.Graph(); G_init.add_nodes_from(range(n))
    G_init.add_edges_from(frozen_edges); G_init.add_edges_from(chosen)
    assert degree_sequence(G_init) == deg_target, "Degree sequence mismatch after projection."

    G_final = anneal_to_target_clustering(
        G_init, target_C=target_clustering, frozen_edges=frozen_edges,
        steps=anneal_steps, T0=T0, Tf=Tf, rng=rng,
    )
    assert degree_sequence(G_final) == deg_target, "Degrees changed during annealing."
    return G_final


def compute_graph_statistics(G: nx.Graph) -> dict:
    """Compute all graph statistics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    stats = {
        'num_nodes': n,
        'num_edges': m,
        'density': 2 * m / (n * (n - 1)) if n > 1 else 0,
        'clustering_coefficient': nx.transitivity(G),
        'is_connected': nx.is_connected(G),
        'diameter': graph_diameter(G),
        'effective_diameter': effective_diameter(G),
    }
    
    return stats


def save_graph_as_edgelist(G: nx.Graph, filepath: str):
    """Save graph as edge list."""
    with open(filepath, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")


# -------------------------------
# Batch Processing
# -------------------------------

def batch_generate_graphs():
    """Generate all graphs according to specifications."""
    
    # Configuration
    dense_path = "./socfb-Reed98/socfb-Reed98.mtx"
    sparse_path = "./soc-wiki-Vote/soc-wiki-Vote.mtx"
    
    # Fixed parameters
    T0 = 0.1
    Tf = 0.005
    K = 10
    twohop_eta = 0.6
    a2_row_cap = 0.05
    deg_power = 1.0
    
    # Define all configurations
    configs = [
        # Sparse configurations
        ('sparse', 0.1, 1000),
        ('sparse', 0.2, 1000),
        ('sparse', 0.3, 1000),
        ('sparse', 0.4, 1000),
        ('sparse', 0.5, 1000),
        ('sparse', 0.6, 10000),
        ('sparse', 0.7, 10000),
        ('sparse', 0.8, 10000),
        ('sparse', 0.9, 10000),
        ('sparse', 1.0, 10000),
        ('sparse', 0.08, 10000),
        ('sparse', 0.06, 10000),
        ('sparse', 0.03, 10000),
        ('sparse', 0.035, 10000),
        ('sparse', 0.01, 10000),
        # Dense configurations
        ('dense', 0.12, 100000),
        ('dense', 0.21, 100000),
        ('dense', 0.3, 100000),
        ('dense', 0.4, 100000),
        ('dense', 0.5, 100000),
        ('dense', 0.6, 100000),
        ('dense', 0.7, 100000),
        ('dense', 0.8, 100000),
        ('dense', 0.9, 100000),
        ('dense', 1.0, 100000),
        ('dense', 0.068, 100000),
        ('dense', 0.062, 1000000),
        ('dense', 0.032, 1000000),
        ('dense', 0.025, 1000000),
    ]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"generated_graphs_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Load base graphs
    print("Loading base graphs...")
    G_sparse = load_graph_from_mtx(sparse_path)
    G_dense = load_graph_from_mtx(dense_path)
    
    # Store all results for CSV
    all_results = []
    
    # Generate graphs
    num_versions = 10
    seeds = list(range(1, num_versions + 1))
    
    for graph_type, target_clustering, anneal_steps in configs:
        print(f"\n{'='*60}")
        print(f"Processing: {graph_type}, target_clustering={target_clustering}, anneal_steps={anneal_steps}")
        print(f"{'='*60}")
        
        G_base = G_dense if graph_type == 'dense' else G_sparse
        
        for version, seed in enumerate(seeds, 1):
            print(f"  Generating version {version}/10 (seed={seed})...")
            
            try:
                # Generate graph
                G_rewired = rewire_with_hh_and_annealing(
                    G_base=G_base,
                    target_clustering=target_clustering,
                    anneal_steps=anneal_steps,
                    K=K,
                    twohop_eta=twohop_eta,
                    a2_row_cap=a2_row_cap,
                    deg_power=deg_power,
                    seed=seed,
                    T0=T0,
                    Tf=Tf,
                )
                
                # Compute statistics
                stats = compute_graph_statistics(G_rewired)
                
                # Create filename
                filename_base = f"semi_synthetic_{graph_type}_clustering_{target_clustering}_v{version}"
                
                # Save as edge list
                edgelist_path = output_dir / f"{filename_base}.edgelist"
                save_graph_as_edgelist(G_rewired, str(edgelist_path))
                
                # Store results
                result = {
                    'graph_type': graph_type,
                    'target_clustering': target_clustering,
                    'anneal_steps': anneal_steps,
                    'version': version,
                    'seed': seed,
                    'filename': filename_base,
                    **stats
                }
                all_results.append(result)
                
                print(f"    Saved {filename_base}")
                print(f"    Actual clustering: {stats['clustering_coefficient']:.6f}")
                
            except Exception as e:
                print(f"    ✗ Error generating version {version}: {e}")
                continue
    
    # Save detailed CSV
    print(f"\n{'='*60}")
    print("Saving detailed statistics CSV...")
    csv_path = output_dir / f"graph_statistics_detailed_{timestamp}.csv"
    fieldnames = ['graph_type', 'target_clustering', 'anneal_steps', 'version', 'seed', 'filename',
                  'num_nodes', 'num_edges', 'density', 'clustering_coefficient', 
                  'is_connected', 'diameter', 'effective_diameter']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Saved detailed statistics to {csv_path}")
    
    # Compute and save summary statistics
    print("Computing summary statistics...")
    summary_stats = []
    
    # Group by configuration
    from itertools import groupby
    all_results_sorted = sorted(all_results, key=lambda x: (x['graph_type'], x['target_clustering'], x['anneal_steps']))
    
    for key, group in groupby(all_results_sorted, key=lambda x: (x['graph_type'], x['target_clustering'], x['anneal_steps'])):
        group_list = list(group)
        graph_type, target_clustering, anneal_steps = key
        
        # Extract statistics
        clustering_vals = [g['clustering_coefficient'] for g in group_list]
        density_vals = [g['density'] for g in group_list]
        diameter_vals = [g['diameter'] for g in group_list]
        eff_diameter_vals = [g['effective_diameter'] for g in group_list]
        
        summary = {
            'graph_type': graph_type,
            'target_clustering': target_clustering,
            'anneal_steps': anneal_steps,
            'num_versions': len(group_list),
            'clustering_mean': np.mean(clustering_vals),
            'clustering_std': np.std(clustering_vals),
            'density_mean': np.mean(density_vals),
            'density_std': np.std(density_vals),
            'diameter_mean': np.mean(diameter_vals),
            'diameter_std': np.std(diameter_vals),
            'effective_diameter_mean': np.mean(eff_diameter_vals),
            'effective_diameter_std': np.std(eff_diameter_vals),
        }
        summary_stats.append(summary)
    
    # Save summary CSV
    summary_csv_path = output_dir / f"graph_statistics_summary_{timestamp}.csv"
    summary_fieldnames = ['graph_type', 'target_clustering', 'anneal_steps', 'num_versions',
                         'clustering_mean', 'clustering_std', 
                         'density_mean', 'density_std',
                         'diameter_mean', 'diameter_std',
                         'effective_diameter_mean', 'effective_diameter_std']
    
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_stats)
    
    print(f"Saved summary statistics to {summary_csv_path}")
    print(f"\n{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    batch_generate_graphs()