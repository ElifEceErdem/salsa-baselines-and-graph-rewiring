import argparse
import os
import math
import random
from collections import defaultdict
import networkx as nx
import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix, csr_matrix
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple
from typing import Optional, Dict, Set, Tuple
from networkx.algorithms import approximation as nx_app
import scipy.sparse as sp  # add this



# -------------------------------
# 0) Utilities
# -------------------------------

def _largest_cc(G):
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()

def graph_diameter(G, approx_ok=True):
    """
    Diameter of the largest connected component.
    If approx_ok, uses NetworkX 2-sweep approximation (fast, good on large graphs).
    Falls back to exact if approximation fails.
    """
    H = _largest_cc(G)
    if H.number_of_nodes() <= 1:
        return 0
    if approx_ok:
        try:
            return nx_app.diameter(H)  # 2-sweep heuristic
        except Exception:
            pass
    return nx.diameter(H)  # exact (can be slow on big graphs)

def effective_diameter(G, q=0.9, k=200, seed=42):
    """
    q-quantile of shortest path lengths (a.k.a. 'effective diameter'), estimated by BFS
    from k random sources on the largest CC. Keeps self-distance=0 out of the pool.
    """
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
        dists.extend([d for v, d in lengths.items() if d > 0])
    if not dists:
        return 0.0
    return float(np.quantile(dists, q))


# ---------- loader ----------
def load_graph_from_mtx(path: str) -> nx.Graph:
    """
    Loads an undirected, simple graph from a .mtx (Matrix Market) file.
    - Symmetrizes A
    - Drops self-loops
    - Binarizes multiple edges
    - Relabels nodes to 0..n-1 (NetworkX default)
    """
    A = mmread(path)  # returns scipy sparse matrix
    if not isinstance(A, (coo_matrix, csr_matrix)):
        A = coo_matrix(A)
    A = A.tocsr()
    # make symmetric
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    # binarize
    A.data[:] = 1
    # networkx graph
    G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
    # ensure contiguous integer labels 0..n-1
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def degree_sequence(G):
    # returns dict node->degree (sorted by node id)
    return {u: G.degree(u) for u in G.nodes()}

def num_triplets_from_degrees(deg):
    # sum_i C(deg_i, 2)
    return sum(d*(d-1)//2 for d in deg.values())

def global_clustering_from_triangles_and_triplets(triangles, triplets, eps=1e-12):
    # standard global clustering coefficient: 3 * triangles / #triplets
    # but with our counting convention (triangles already "triangle count"),
    # C = 3T / triplets. However many definitions use T / triplets_of_nodes;
    # in edge-swap annealing we only need consistency, not the factor.
    return (3.0 * triangles) / (triplets + eps)

def common_neighbors_fast(N, u, v):
    # N is dict: node -> set(neighbors)
    # returns |N(u) ∩ N(v)|
    if len(N[u]) > len(N[v]):
        u, v = v, u
    return sum((w in N[v]) for w in N[u])

def triangles_count(G):
    # exact triangles via NetworkX (sum of node-triangles / 3)
    tri_by_node = nx.triangles(G)
    return sum(tri_by_node.values()) // 3

def make_centered_scorer(
    Z: np.ndarray,
    degrees: Optional[Dict[int, int]] = None,
    center: str = "degree",      # "none" | "mean" | "degree"
    deg_power: float = 1.0,
    use_cosine: bool = True
):
    """
    Returns score(u,v) on centered embeddings:
      - center="none": no centering
      - center="mean": subtract global mean
      - center="degree": subtract degree-weighted mean (weights ~ deg^deg_power)
    If use_cosine=True => cosine similarity; else plain dot after centering.
    """
    Z = np.asarray(Z, dtype=np.float32)
    n, d = Z.shape

    if center == "none":
        Zc = Z
    elif center == "mean":
        m = Z.mean(axis=0, keepdims=True)
        Zc = Z - m
    elif center == "degree":
        if degrees is None:
            raise ValueError("degrees dict required for center='degree'")
        deg = np.array([degrees[i] for i in range(n)], dtype=np.float32)
        w = np.power(np.maximum(deg, 0.0), deg_power)
        w = w / (w.sum() + 1e-12)
        m = (w[:, None] * Z).sum(axis=0, keepdims=True)
        Zc = Z - m
    else:
        raise ValueError(f"Unknown center mode: {center}")

    if use_cosine:
        norms = np.linalg.norm(Zc, axis=1) + 1e-12
        def score(u, v):
            a = Zc[u]; b = Zc[v]
            return float((a @ b) / (norms[u] * norms[v]))
    else:
        def score(u, v):
            return float(Zc[u] @ Zc[v])

    return score


# -------------------------------
# 1) Encoder & scoring (swap with your GAE)
# -------------------------------

def simple_sgc_encoder(
    G,
    X=None,
    K=2,
    out_dim=64,
    seed=42,
    normalize_rows=True,
    twohop_eta: float = 0.0,        # NEW: weight for A^2 skip (0.0 = off)
    add_self_loops: bool = True,    # like standard SGC
    a2_row_cap: float = None,       # NEW: optional cap on row-normalized A^2 weights (e.g., 0.05)
):
    """
    Z = ( Ahat  +  twohop_eta * RowNorm(A^2) )^K  X  W
      where Ahat = D^{-1/2}(A + I)^? D^{-1/2}  (if add_self_loops=True)
    Notes:
      - A is the (0/1) adjacency without self-loops.
      - RowNorm(A^2) emphasizes wedge closure (two-hop paths).
      - Optional a2_row_cap clips large row-normalized entries to reduce hub bias.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    order = list(range(n))
    A = nx.to_scipy_sparse_array(G, nodelist=order, format="csr")
    A = A.tocsr()
    A.setdiag(0); A.eliminate_zeros()

    # Base normalized adjacency (with or without self-loops)
    if add_self_loops:
        A_sl = (A + sp.eye(n, format="csr"))
    else:
        A_sl = A

    deg = np.array(A_sl.sum(1)).ravel()
    deg_safe = np.maximum(deg, 1e-12)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_safe))
    Ahat = D_inv_sqrt @ A_sl @ D_inv_sqrt

    # Optional two-hop term
    if twohop_eta > 0.0:
        A2 = (A @ A).tocsr()
        A2.setdiag(0); A2.eliminate_zeros()
        rowsum = np.maximum(np.asarray(A2.sum(1)).ravel(), 1e-12)

        B = A2.multiply(1.0 / rowsum[:, None])  # RowNorm(A^2)

        if a2_row_cap is not None:
            # Clip large transition weights per row (helps dense hubby graphs)
            B_coo = B.tocoo(copy=True)
            data = B_coo.data
            data = np.minimum(data, float(a2_row_cap))
            B = sp.csr_matrix((data, (B_coo.row, B_coo.col)), shape=B.shape)

        F = (Ahat + twohop_eta * B).tocsr()
        # Optional: symmetrize to stay undirected
        F = 0.5 * (F + F.T)
    else:
        F = Ahat

    # Features
    if X is None:
        X = np.eye(n, dtype=np.float32)  # super-stable, no feature engineering needed

    # K-hop propagation
    H = X
    for _ in range(K):
        H = F @ H

    # Linear projection + (optional) row-normalization
    W = rng.standard_normal((H.shape[1], out_dim)) / math.sqrt(H.shape[1])
    Z = H @ W
    if normalize_rows:
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return Z



def edge_scores_from_embeddings(Z, nodes=None):
    """
    Returns a function score(u, v) = Z[u]·Z[v].
    Avoid forming the full dense matrix for huge graphs; we’ll
    call this only for needed pairs.
    """
    if nodes is None:
        nodes = np.arange(Z.shape[0])
    Z = np.asarray(Z, dtype=np.float32)
    def score(u, v):  # dot product
        return float(np.dot(Z[u], Z[v]))
    return score

def make_sgc_tri_score(G: nx.Graph, Z: np.ndarray, alpha: float = 1.0, beta: float = 0.0):
    """
    SGC-Tri score:
      s(u,v) = (Z_u · Z_v) + alpha * CN(u,v)/sqrt(d(u)d(v)) + beta * Jaccard(u,v)
    Returns (score_fn, N0) where:
      - score_fn(u,v) gives the weight
      - N0 is {node -> set(neighbors)} on the BASE graph (for optional reuse)
    """
    N0 = {u: set(G.neighbors(u)) for u in G.nodes()}
    deg = {u: len(N0[u]) for u in G.nodes()}

    def jaccard(u, v):
        inter = common_neighbors_fast(N0, u, v)
        den = len(N0[u]) + len(N0[v]) - inter
        return (inter / den) if den else 0.0

    def score(u, v):
        base = float(np.dot(Z[u], Z[v]))  # your SGC dot
        cn = common_neighbors_fast(N0, u, v)
        salton = cn / math.sqrt(max(deg[u], 1) * max(deg[v], 1))
        return base + alpha * salton + beta * jaccard(u, v)

    return score, N0


# -------------------------------
# 2) Candidate pruning
# -------------------------------

def build_candidate_edges(G, score_fn, L=128, add_triangle_closing=True, cn_per_node=32, rng=None):
    """
    For each node, keep top-L non-neighbors by score.
    Optionally add up to cn_per_node triangle-closing pairs from 2-hop neighborhoods.
    Returns: dict {u: list of candidate neighbors v>u} and a weight dict w[(u,v)].
    """
    if rng is None:
        rng = random.Random(0)

    n = G.number_of_nodes()
    nodes = list(range(n))
    neighbor_sets = {u: set(G.neighbors(u)) for u in nodes}

    # Top-L by dot-product (naive O(n^2) if used literally; for big graphs,
    # compute top-L via ANN/blocks. Here we do a simple filtered scan.)
    cand = defaultdict(list)
    weight = {}

    # Light heuristic: only score within a random subset if graph is huge
    # (you can replace this with FAISS/Annoy for real scale)
    for u in nodes:
        # exclude self and current neighbors
        forbidden = neighbor_sets[u] | {u}
        # naive: score against all, then take top-L of allowed
        scores = []
        for v in nodes:
            if v in forbidden: 
                continue
            s = score_fn(u, v)
            scores.append((s, v))
        scores.sort(reverse=True)
        for _, v in scores[:L]:
            a, b = (u, v) if u < v else (v, u)
            if b not in cand[a]:
                cand[a].append(b)
                weight[(a, b)] = score_fn(a, b)

    if add_triangle_closing:
        # add candidates from 2-hop neighborhoods
        for u in nodes:
            two_hop = set()
            for w in neighbor_sets[u]:
                two_hop.update(neighbor_sets[w])
            two_hop -= neighbor_sets[u]
            two_hop.discard(u)
            # pick up to cn_per_node with highest CN
            cn_scores = []
            for v in two_hop:
                if v in neighbor_sets[u]: 
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a in cand) and (b in cand[a]): 
                    continue
                # CN score (no need to be exact here)
                cn = len(neighbor_sets[u] & neighbor_sets[v])
                if cn > 0:
                    cn_scores.append((cn, v))
            cn_scores.sort(reverse=True)
            for _, v in cn_scores[:cn_per_node]:
                a, b = (u, v) if u < v else (v, u)
                if b not in cand[a]:
                    cand[a].append(b)
                    # blend dot score + lambda*CN
                    base = score_fn(a, b)
                    weight[(a, b)] = base  # we’ll add lambda*CN later if desired

    # ensure dedup + sort for stability
    for a in cand:
        cand[a] = sorted(set(cand[a]))
    return cand, weight

# -------------------------------
# 3) Spanning tree backbone (optional)
# -------------------------------

def maximum_weight_spanning_tree_from_candidates(n, cand, weight):
    """
    Build a weighted graph from candidates and take maximum spanning tree.
    Returns set of frozen edges {(u,v)} and degree tally deg_T.
    """
    H = nx.Graph()
    H.add_nodes_from(range(n))
    for a, nbrs in cand.items():
        for b in nbrs:
            w = weight.get((a, b), 0.0)
            H.add_edge(a, b, weight=w)
    T = nx.maximum_spanning_tree(H, weight="weight")
    frozen = set(tuple(sorted(e)) for e in T.edges())
    deg_T = defaultdict(int)
    for u, v in frozen:
        deg_T[u] += 1
        deg_T[v] += 1
    return frozen, deg_T

def spanning_tree_from_base_graph(G_base, score_fn=None):
    """
    Build a spanning tree *only from base edges* to ensure deg_T(i) ≤ deg_base(i).
    If score_fn is provided, use it as edge weight; otherwise unweighted MST.
    Returns: frozen_edges (set of (u,v)), deg_T (dict)
    """
    H = nx.Graph()
    H.add_nodes_from(G_base.nodes())

    if score_fn is None:
        # unweighted: just take any spanning tree
        T = nx.minimum_spanning_tree(G_base)  # works the same as any spanning tree on unweighted
    else:
        # weighted by encoder score on existing edges
        for u, v in G_base.edges():
            w = score_fn(u, v)
            H.add_edge(u, v, weight=w)
        T = nx.maximum_spanning_tree(H, weight="weight")

    frozen = set(tuple(sorted(e)) for e in T.edges())
    deg_T = defaultdict(int)
    for u, v in frozen:
        deg_T[u] += 1
        deg_T[v] += 1
    return frozen, deg_T


# -------------------------------
# 4) Weighted Havel–Hakimi (projection)
# -------------------------------

def ensure_feasible_candidates(
    n: int,
    G_base: nx.Graph,
    cand: dict[int, list[int]],
    weight: dict[tuple[int,int], float],
    residual_deg: dict[int, int],
    frozen_edges: set[tuple[int,int]],
    score_fn,
    L_extra_step: int = 64,
    max_expansions: int = 5,
):
    """
    Ensure each node has >= residual_deg[u] available candidate neighbors (excluding frozen edges).
    Adds (a) base non-tree edges, then (b) additional top-K scored nodes as needed.
    """
    N_base = {u: set(G_base.neighbors(u)) for u in range(n)}
    frozen = set(frozen_edges)

    def add_candidate(u, v):
        a, b = (u, v) if u < v else (v, u)
        if a == b: 
            return
        if (a, b) in frozen: 
            return
        # add both directions in cand (we store with a<b in weight)
        cand.setdefault(a, [])
        if b not in cand[a]:
            cand[a].append(b)
        cand.setdefault(b, [])
        if a not in cand[b]:
            cand[b].append(a)
        if (a, b) not in weight:
            weight[(a, b)] = score_fn(a, b)

    # Step A: add all remaining base edges (non-tree) as candidates
    for u, v in G_base.edges():
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in frozen:
            continue
        add_candidate(a, b)

    # Step B: expand with more high-score pairs until capacity per node >= demand
    nodes = list(range(n))
    for _ in range(max_expansions):
        deficit = []
        for u in nodes:
            if residual_deg[u] <= 0:
                continue
            # available = candidate neighbors not frozen and not identical
            avail_u = 0
            for v in cand.get(u, []):
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in frozen:
                    continue
                avail_u += 1
            need = residual_deg[u] - avail_u
            if need > 0:
                deficit.append((need, u))
        if not deficit:
            break  # feasible wrt capacities

        # Expand candidates for the worst-off nodes
        deficit.sort(reverse=True)  # address biggest deficits first
        for _, u in deficit:
            forbidden = set(cand.get(u, [])) | {u}
            # also forbid frozen endpoints
            for v in range(n):
                if v in forbidden:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in frozen:
                    continue
                add_candidate(u, v)
                # stop early; we bulk-add more below via top-K
            # Add extra top-K by score (bulk)
            scores = []
            for v in range(n):
                if v == u:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in frozen:
                    continue
                if v in cand.get(u, []):
                    continue
                scores.append((score_fn(u, v), v))
            scores.sort(reverse=True)
            for _, v in scores[:L_extra_step]:
                add_candidate(u, v)
    # Add at the END of ensure_feasible_candidates(...), just before 'return cand, weight'
    # Final safety net: if any node still under-capacitated, open complete graph (minus frozen)
    cap_per_node = {
        u: sum(1 for v in cand.get(u, []) if (min(u,v), max(u,v)) not in frozen_edges)
        for u in range(n)
    }
    under = [u for u in range(n) if residual_deg[u] > cap_per_node.get(u, 0)]
    if under:
        for u in under:
            for v in range(n):
                if v == u:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in frozen_edges:
                    continue
                # add if missing
                if v not in cand.get(u, []):
                    cand.setdefault(u, []).append(v)
                    weight[(a, b)] = weight.get((a, b), score_fn(a, b))
                if u not in cand.get(v, []):
                    cand.setdefault(v, []).append(u)

    return cand, weight

def weighted_havel_hakimi(n, residual_deg, cand, weight, forbidden_edges=frozenset(), lambda_cn=0.0, neighbor_sets=None):
    """
    Choosy HH on a pruned candidate graph:
      - Always returns exactly the requested degree sequence IF feasible on cand.
      - Uses weights to pick the best neighbors first.
      - 'forbidden_edges' (frozen) cannot be touched/duplicated.
      - Optionally add lambda_cn * CN(u,v) to weight (requires neighbor_sets from CURRENT partial graph).
    Returns: set of chosen edges S (as (u,v) u<v), or raises ValueError if infeasible.
    """
    # Work on copies
    rem = residual_deg.copy()
    S = set()
    # Build availability lists
    avail = {u: set() for u in range(n)}
    for a, nbrs in cand.items():
        for b in nbrs:
            if (a, b) not in forbidden_edges:
                avail[a].add(b)
                avail.setdefault(b, set()).add(a)

    # Helper for effective weight
    def eff_w(u, v):
        base = weight.get((min(u,v), max(u,v)), 0.0)
        if lambda_cn != 0.0 and neighbor_sets is not None:
            cn = common_neighbors_fast(neighbor_sets, u, v)
            return base + lambda_cn * cn
        return base

    # Priority: nodes with largest remaining degree
    # Simple heuristic loop with limited backtracking
    stack = []
    while True:
        # find node with largest residual degree
        u = max(rem, key=lambda x: rem[x])
        if rem[u] == 0:
            # done?
            if all(d == 0 for d in rem.values()):
                break
            # there may be others with >0; continue
            candidates_left = [x for x in rem if rem[x] > 0]
            if not candidates_left:
                break
            u = max(candidates_left, key=lambda x: rem[x])

        r = rem[u]
        # feasible neighbors: not already saturated, edge allowed, and not already chosen
        feas = []
        for v in avail.get(u, ()):
            if rem.get(v, 0) <= 0:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in S or (a, b) in forbidden_edges:
                continue
            feas.append((eff_w(u, v), v))

        if len(feas) < r:
            # basic backtrack: if impossible on current cand, fail
            raise ValueError("Residual degree sequence not realizable on candidate set.")

        # pick top-r by weight
        feas.sort(reverse=True)
        chosen_vs = [v for _, v in feas[:r]]

        # commit edges (u, chosen)
        for v in chosen_vs:
            a, b = (u, v) if u < v else (v, u)
            S.add((a, b))
            rem[u] -= 1
            rem[v] -= 1

            # Update availability (optional; not strictly needed)
            # We leave avail as-is; rem checks prevent overuse.

    return S

# -------------------------------
# 5) Degree-preserving edge-swap annealing
# -------------------------------
# working, but hits clustering ceiling (can just double)
def anneal_to_target_clustering(G, target_C, frozen_edges=frozenset(), steps=50_000, T0=1.0, Tf=0.01, rng=None):
    """
    Degree-preserving edge swaps to move clustering to target_C.
    Frozen edges are never modified.
    Uses incremental triangle count; degrees (and triplets) remain constant.
    """
    if rng is None:
        rng = random.Random(0)

    n = G.number_of_nodes()
    edges = set(tuple(sorted(e)) for e in G.edges())
    nonfrozen_edges = [e for e in edges if e not in frozen_edges]
    N = {u: set(G.neighbors(u)) for u in G.nodes()}

    # Precompute triangles and triplets
    T = triangles_count(G)
    triplets = num_triplets_from_degrees(degree_sequence(G))
    target_T = target_C * triplets / 3.0  # since C = 3T / triplets

    def edge_triangle(u, v):
        return common_neighbors_fast(N, u, v)

    def accept(oldS, newS, temp):
        if newS < oldS:
            return True
        # SA acceptance
        return rng.random() < math.exp(-(newS - oldS) / max(temp, 1e-9))

    def propose_swap():
        # sample two disjoint, non-frozen edges (a,b) and (c,d)
        (a, b) = rng.choice(nonfrozen_edges)
        (c, d) = rng.choice(nonfrozen_edges)
        if len({a, b, c, d}) < 4:
            return None
        # two possible rewirings
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
        # Try candidate 1 then 2
        for x, y in ((e1, e2), (f1, f2)):
            # validity: edges must not already exist, must not include frozen
            if x in edges or y in edges: 
                continue
            if x in frozen_edges or y in frozen_edges:
                continue
            u1, v1 = x
            u2, v2 = y
            # compute Δ triangles: add new - remove old
            delta = (edge_triangle(*x) + edge_triangle(*y)
                     - edge_triangle(a, b) - edge_triangle(c, d))
            newT = T + delta
            newS = abs(newT - target_T)
            if accept(S, newS, temp):
                # commit swap
                # remove old edges
                edges.remove((a, b)); edges.remove((c, d))
                nonfrozen_edges.remove((a, b)); nonfrozen_edges.remove((c, d))
                # update neighbor sets
                N[a].remove(b); N[b].remove(a)
                N[c].remove(d); N[d].remove(c)
                # add new edges
                edges.add(x); edges.add(y)
                nonfrozen_edges.append(x); nonfrozen_edges.append(y)
                N[u1].add(v1); N[v1].add(u1)
                N[u2].add(v2); N[v2].add(u2)
                # update triangle count and score
                T = newT
                S = newS
                break  # go to next step

    # rebuild graph from final edge set
    H = nx.Graph()
    H.add_nodes_from(range(n))
    H.add_edges_from(edges)
    return H


# -------------------------------
# 6) Orchestrator
# -------------------------------

def project_bmatching_min_cost_flow(
    n: int,
    residual_deg: dict[int,int],
    cand: dict[int, list[int]],
    weight: dict[tuple[int,int], float],
    frozen_edges: set[tuple[int,int]],
):
    """
    Exact projection via min-cost flow with 1-edge-per-pair cap:
      - Nodes are split into L (supply) and R (demand).
      - For each unordered candidate pair {u,v} (u<v) not frozen, add ONE arc L_u -> R_v
        with capacity 1 and cost = -weight(u,v).
      - Supplies on L_u = residual_deg[u]; Demands on R_v = residual_deg[v].
    This prevents sending two units through the same {u,v}.
    """

    # Build flow network
    F = nx.DiGraph()
    src, sink = "_src", "_sink"
    F.add_node(src)
    F.add_node(sink)

    L = {u: f"L{u}" for u in range(n)}
    R = {u: f"R{u}" for u in range(n)}

    total_stubs = sum(residual_deg.values())
    if total_stubs % 2 != 0:
        raise ValueError("Residual degree sum must be even.")

    # Source to L (supplies)
    for u in range(n):
        d = residual_deg[u]
        if d > 0:
            F.add_edge(src, L[u], capacity=d, weight=0)

    # R to sink (demands)
    for v in range(n):
        d = residual_deg[v]
        if d > 0:
            F.add_edge(R[v], sink, capacity=d, weight=0)

    # Candidate arcs: ONLY one direction per unordered pair (u < v)
    scale = 1e6  # scale float weights to ints
    seen_pairs = set()
    for a, nbrs in cand.items():
        for b in nbrs:
            u, v = (a, b) if a < b else (b, a)
            if u == v:
                continue
            if (u, v) in seen_pairs:
                continue  # already added
            seen_pairs.add((u, v))
            if (u, v) in frozen_edges:
                continue
            w = weight.get((u, v), 0.0)
            F.add_edge(L[u], R[v], capacity=1, weight=int(round(-scale * w)))

    # Compute min-cost flow that satisfies all supplies/demands
    flow = nx.algorithms.flow.min_cost_flow(F)

    # Extract chosen edges
    chosen = set()
    deg_used = defaultdict(int)

    for u in range(n):
        Lu = L[u]
        for rv, f in flow.get(Lu, {}).items():
            if f <= 0 or not rv.startswith("R"):
                continue
            v = int(rv[1:])
            # this corresponds to the unordered pair (u, v) with u < v?
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in frozen_edges:
                continue
            if f > 1:
                # Shouldn't happen due to capacity=1, but guard anyway
                f = 1
            # add exactly one edge
            chosen.add((a, b))
            deg_used[u] += 1
            deg_used[v] += 1

    # Verify degrees match residual demands
    for u in range(n):
        if deg_used.get(u, 0) != residual_deg[u]:
            raise ValueError(
                f"Flow projection did not realize residual degrees for node {u}: "
                f"used {deg_used.get(u,0)} vs need {residual_deg[u]}"
            )

    return chosen

def build_full_candidate_set(
    n: int,
    frozen_edges: set[tuple[int,int]],
    score_fn,
    lambda_cn: float = 0.0,
    N0: Optional[Dict[int, Set[int]]] = None,
):
    """
    Return (cand, weight) where cand has ALL unordered pairs (u,v), u!=v,
    except those in frozen_edges. Weight[(u,v)] = score_fn(u,v) [+ lambda_cn*CN].
    cand[u] contains all v != u (symmetrized).
    """
    cand = {u: [] for u in range(n)}
    weight = {}

    # Optional CN bias needs neighbor sets on the BASE graph
    use_cn = (lambda_cn != 0.0 and N0 is not None)

    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in frozen_edges:
                continue
            cand[u].append(v)
            cand.setdefault(v, []).append(u)
            w = score_fn(u, v)
            if use_cn:
                cn = len(N0[u] & N0[v])
                w = w + lambda_cn * cn
            weight[(u, v)] = w

    # sort for determinism
    for u in range(n):
        cand[u].sort()
    return cand, weight

def rewire_with_hh_and_annealing(
    G_base: nx.Graph,
    target_clustering: float,
    use_spanning_tree: bool = True,
    use_full_candidates: bool = True,   # full candidate graph (minus frozen edges) by default
    L: int = 128,                       # used only if use_full_candidates=False
    cn_per_node: int = 32,              # used only if use_full_candidates=False
    anneal_steps: int = 50_000,
    seed: int = 42,
    T0: float = 1.5,
    Tf: float = 0.005,
    tri_alpha: Optional[float] = None,
    tri_beta: float = 0.0,
    # legacy: allow passing --lambda_cn; if tri_alpha is None, we’ll use this
    lambda_cn: float = 0.5,
):
    """
    Pipeline:
      1) Encode & score pairs with your encoder (GAE/SGC/etc.)
      2) (Optional) Freeze a spanning tree built from BASE edges (keeps connectivity, safe degrees)
      3) Compute residual degrees: deg_target - deg_tree
      4) Build candidate set:
            - full candidates: all (u,v), u<v, except frozen edges
            - or pruned candidates (top-L + triangle-closing), then ensure feasibility
      5) Project exactly to residual degrees:
            - try Weighted Havel–Hakimi on candidate graph
            - fallback to min-cost flow b-matching
      6) Combine frozen + chosen edges -> G_init
      7) Degree-preserving edge-swap annealing to reach target clustering (tree edges frozen)
    Guarantees:
      - Degree sequence (and density) exactly preserved
      - Connectivity preserved if use_spanning_tree=True
    """
    rng = random.Random(seed)
    n = G_base.number_of_nodes()
    nodes = list(range(n))

    # 1) Encode & scoring (swap stub with your GAE)
    if getattr(args, "use_cosine", False):
        # Cosine likes unnormalized rows
        Z = simple_sgc_encoder(
            G_base, out_dim=64, seed=seed,
            normalize_rows=False,
            K=getattr(args, "K", 2),
            twohop_eta=getattr(args, "twohop_eta", 0.0),
            a2_row_cap=getattr(args, "a2_row_cap", None),
        )
        score_fn = make_centered_scorer(
            Z,
            degrees=dict(G_base.degree()),
            center=getattr(args, "center", "degree"),
            deg_power=getattr(args, "deg_power", 1.0),
            use_cosine=True,
        )
        N0 = {u: set(G_base.neighbors(u)) for u in G_base.nodes()}
    else:
        # SGC-Tri path (works fine with A^2 too)
        Z = simple_sgc_encoder(
            G_base, out_dim=64, seed=seed,
            normalize_rows=True,
            K=getattr(args, "K", 2),
            twohop_eta=getattr(args, "twohop_eta", 0.0),
            a2_row_cap=getattr(args, "a2_row_cap", None),
        )
        alpha_cn = tri_alpha if (tri_alpha is not None) else lambda_cn
        score_fn, N0 = make_sgc_tri_score(
            G_base, Z,
            alpha=float(alpha_cn),
            beta=float(tri_beta)
        )



    # 2) Spanning tree from BASE graph (ensures deg_T[i] ≤ deg_base[i])
    if use_spanning_tree:
        frozen_edges, deg_T = spanning_tree_from_base_graph(G_base, score_fn=score_fn)
    else:
        frozen_edges, deg_T = set(), defaultdict(int)

    # 3) Residual degrees
    deg_target = degree_sequence(G_base)
    residual_deg = {u: deg_target[u] - deg_T.get(u, 0) for u in nodes}
    if any(d < 0 for d in residual_deg.values()):
        raise ValueError("Frozen edges exceed target degrees for some nodes.")

    # 4) Candidate set
    if use_full_candidates:
        # Complete graph minus frozen edges
        # cand, weight = build_full_candidate_set(n, frozen_edges, score_fn)
        # lambda_cn=0.0 so we don’t add CN twice; pass N0 for completeness (not used when lambda_cn=0)
        cand, weight = build_full_candidate_set(n, frozen_edges, score_fn, lambda_cn=0.0, N0=N0)

    else:
        # Pruned candidate graph + feasibility expansion
        cand, weight = build_candidate_edges(
            G_base, score_fn, L=L, add_triangle_closing=True, cn_per_node=cn_per_node
        )
        cand, weight = ensure_feasible_candidates(
            n=n,
            G_base=G_base,
            cand=cand,
            weight=weight,
            residual_deg=residual_deg,
            frozen_edges=frozen_edges,
            score_fn=score_fn,
            L_extra_step=max(64, L),  # expand generously
            max_expansions=6,
        )

    # 5) Exact projection: try HH, fallback to min-cost flow
    try:
        chosen = weighted_havel_hakimi(
            n=n,
            residual_deg=residual_deg,
            cand=cand,
            weight=weight,
            forbidden_edges=frozen_edges,
            lambda_cn=0.0,         # weight already blended outside if desired
            neighbor_sets=None
        )
    except ValueError:
        print("HH failed on the candidate graph; falling back to min-cost flow b-matching...")
        chosen = project_bmatching_min_cost_flow(
            n=n,
            residual_deg=residual_deg,
            cand=cand,
            weight=weight,
            frozen_edges=frozen_edges
        )

    # 6) Build initial graph (frozen + chosen)
    G_init = nx.Graph()
    G_init.add_nodes_from(range(n))
    G_init.add_edges_from(frozen_edges)
    G_init.add_edges_from(chosen)

    # Sanity: exact degree match (=> exact density)
    assert degree_sequence(G_init) == deg_target, "Degree sequence mismatch after projection."

    # 7) Anneal to target clustering with degree-preserving swaps; freeze tree edges
    G_final = anneal_to_target_clustering(
        G_init,
        target_C=target_clustering,
        frozen_edges=frozen_edges,
        steps=anneal_steps,
        T0=T0,
        Tf=Tf,
        rng=rng
    )

    # Final sanity
    assert degree_sequence(G_final) == deg_target, "Degrees changed during annealing."

    return G_final


## ----- plotting_utils ---- ###
def _degree_hist_aligned(graphs: List[nx.Graph]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Build a single, aligned degree axis and per-graph frequency arrays.
    Returns:
      degrees_axis: np.array of all degree values present across graphs (sorted)
      counts_list: list of frequency arrays aligned to degrees_axis
    """
    # collect all degrees across graphs
    all_deg_values = set()
    per_counts = []
    for G in graphs:
        degs = [d for _, d in G.degree()]
        cnt = Counter(degs)
        per_counts.append(cnt)
        all_deg_values.update(cnt.keys())
    degrees_axis = np.array(sorted(all_deg_values))
    counts_list = []
    for cnt in per_counts:
        counts_list.append(np.array([cnt.get(k, 0) for k in degrees_axis], dtype=int))
    return degrees_axis, counts_list

def plot_degree_distribution(graphs, labels, filename, styles=None):
    """
    styles: optional list like ['-', (0,(2,2))] for solid & dotted.
    """

    if styles is None:
        # solid, dotted, dashdot, dashed... (cycle as needed)
        styles = ['-', (0, (2, 2)), (0, (3, 1, 1, 1)), '--', '-.']

    # precompute aligned hist bins
    degrees_hist = []
    all_deg = set()
    for G in graphs:
        d = [deg for _, deg in G.degree()]
        all_deg.update(d)
    x = np.array(sorted(all_deg))
    for G in graphs:
        d = [deg for _, deg in G.degree()]
        counts = np.bincount(d, minlength=x.max()+1)[x]
        degrees_hist.append((x, counts))

    os.makedirs("plots_selected_10version", exist_ok=True)

    # log-log
    plt.figure(figsize=(8, 6))
    for i, ((bins, hist), label) in enumerate(zip(degrees_hist, labels)):
        plt.plot(bins[bins>0], hist[bins>0],
                 label=label,
                 linestyle=styles[i % len(styles)],
                 linewidth=2,
                 alpha=0.95,
                 zorder=2+i)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Degree"); plt.ylabel("Frequency")
    plt.title("Degree Distributions (log-log)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots_selected_10version/{filename}_loglog.png", dpi=300)
    plt.close()

    # linear-linear
    plt.figure(figsize=(8, 6))
    for i, ((bins, hist), label) in enumerate(zip(degrees_hist, labels)):
        plt.plot(bins, hist,
                 label=label,
                 linestyle=styles[i % len(styles)],
                 linewidth=2,
                 alpha=0.95,
                 zorder=2+i)
    plt.xlabel("Degree"); plt.ylabel("Frequency")
    plt.title("Degree Distributions (linear)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots_selected_10version/{filename}_linear.png", dpi=300)
    plt.close()

def plot_degree_cdf(G1, G2, labels, filename, styles=None):
    if styles is None:
        styles = ['-', (0, (2, 2))]  # solid, dotted

    plt.figure(figsize=(8, 6))
    for i, (G, label) in enumerate(zip([G1, G2], labels)):
        degrees = np.sort([d for _, d in G.degree()])
        cdf = np.arange(1, len(degrees) + 1) / len(degrees)
        mask = degrees > 0  # avoid log(0) on x-axis
        plt.plot(degrees[mask], cdf[mask],
                 label=label,
                 linestyle=styles[i % len(styles)],
                 linewidth=2,
                 alpha=0.95,
                 zorder=2+i)
    plt.xscale("log")
    plt.xlabel("Degree"); plt.ylabel("CDF")
    plt.title("Degree Distribution (CDF, log-x)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    os.makedirs("plots_selected_10version", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots_selected_10version/{filename}", dpi=300)
    plt.close()

def plot_graph(G: nx.Graph, title: str = "Graph", node_size: int = 20,
               edge_alpha: float = 0.08, save_path: str = None, seed: int = 42,
               max_nodes_to_draw: int = 3000) -> None:
    """
    Draws a graph (downsamples if very large to avoid slow layouts).
    Uses spring layout; for n>max_nodes_to_draw, draws an induced subgraph.
    """
    if G.number_of_nodes() > max_nodes_to_draw:
        # sample nodes proportional to degree for nicer visuals
        degs = np.array([d for _, d in G.degree()], dtype=float)
        probs = (degs + 1e-9) / (degs.sum() + 1e-9)
        sample = np.random.default_rng(seed).choice(G.number_of_nodes(), size=max_nodes_to_draw, replace=False, p=probs)
        H = G.subgraph(sample).copy()
        title = f"{title} (induced subgraph of {max_nodes_to_draw} nodes)"
        G_draw = H
    else:
        G_draw = G

    pos = nx.spring_layout(G_draw, seed=seed)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G_draw, pos, node_size=node_size, alpha=0.9)
    nx.draw_networkx_edges(G_draw, pos, alpha=edge_alpha, width=0.5)
    plt.title(title)
    plt.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    plt.close()

# Optional: a quick numerical check to assert equality of degree sequences
def assert_degree_sequence_match(G1: nx.Graph, G2: nx.Graph) -> None:
    d1 = sorted([d for _, d in G1.degree()])
    d2 = sorted([d for _, d in G2.degree()])
    if d1 != d2:
        # give a short diff summary
        c1, c2 = Counter(d1), Counter(d2)
        allk = sorted(set(c1) | set(c2))
        diffs = [(k, c1.get(k,0), c2.get(k,0)) for k in allk if c1.get(k,0) != c2.get(k,0)]
        raise AssertionError(f"Degree sequence mismatch! First few diffs: {diffs[:10]}")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewire a graph to target clustering with exact degree/density.")
    gsel = parser.add_mutually_exclusive_group(required=True)
    gsel.add_argument("--dense", action="store_true", help="Use the dense baseline graph path")
    gsel.add_argument("--sparse", action="store_true", help="Use the sparse baseline graph path")

    parser.add_argument("--dense_path", type=str,
        default="./socfb-Reed98/socfb-Reed98.mtx",
        help="Path to the dense .mtx file")
    parser.add_argument("--sparse_path", type=str,
        default="./soc-wiki-Vote/soc-wiki-Vote.mtx",
        help="Path to the sparse .mtx file")

    parser.add_argument("--target_clustering", type=float, default=None,
        help="Target global clustering coefficient. If omitted, uses base graph's clustering.")
    parser.add_argument("--L", type=int, default=128, help="Top-L candidates per node")
    parser.add_argument("--lambda_cn", type=float, default=0.5, help="Weight for CN bias in scoring")
    parser.add_argument("--cn_per_node", type=int, default=32, help="Extra triangle-closing candidates per node")
    parser.add_argument("--no_tree", action="store_true", help="Disable spanning-tree connectivity backbone")
    parser.add_argument("--anneal_steps", type=int, default=20000, help="Number of edge-swap steps")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--full_candidates", action="store_true",
    help="Use the full complete graph (minus frozen) as candidate set for projection.")
    parser.add_argument("--T0", type=float, default=1.5, help="Initial temperature for simulated annealing (>= Tf)")
    parser.add_argument("--Tf", type=float, default=0.005, help="Final temperature for simulated annealing")


    parser.add_argument("--save_edgelist", type=str, default=None,
        help="If set, save rewired graph as an edgelist to this path")
    parser.add_argument("--save_mtx", type=str, default=None,
        help="If set, save rewired graph as Matrix Market (.mtx) to this path")
    
    parser.add_argument("--tri_alpha", type=float, default=None,
    help="Weight for CN/salton term in SGC-Tri; if None, fallback to --lambda_cn")
    parser.add_argument("--tri_beta", type=float, default=0.0,
        help="Weight for Jaccard term in SGC-Tri")
    
    # Cosine/centering controls (good for dense graphs)
    parser.add_argument("--use_cosine", action="store_true",
        help="Use centered cosine scorer instead of SGC-Tri/dot")
    parser.add_argument("--center", type=str, default="degree",
        choices=["none", "mean", "degree"],
        help="Centering strategy for embeddings when using --use_cosine")
    parser.add_argument("--deg_power", type=float, default=1.0,
        help="Exponent for degree weights in degree-centered mean (deg^p)")
    
    # Encoder hyperparams
    parser.add_argument("--K", type=int, default=2, help="Propagation depth for encoder")
    parser.add_argument("--twohop_eta", type=float, default=0.0,
        help="Weight for triangle-aware A^2 row-normalized skip in the SGC filter (0=off)")
    parser.add_argument("--a2_row_cap", type=float, default=None,
        help="Optional cap for each row-normalized A^2 entry (e.g., 0.05) to reduce hub bias")


    args = parser.parse_args()

    # 1) Pick graph
    if args.dense:
        graph_path = args.dense_path
        print(f"Loading dense graph from: {graph_path}")
    else:
        graph_path = args.sparse_path
        print(f"Loading sparse graph from: {graph_path}")

    

    G = load_graph_from_mtx(graph_path)
    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    base_density = 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
    base_C = nx.transitivity(G)
    print(f"Base density = {base_density:.6f} | Base global clustering = {base_C:.6f} | Connected? {nx.is_connected(G)}")

    base_diam = graph_diameter(G, approx_ok=True)
    base_eff_diam = effective_diameter(G, q=0.9, k=200)

    print(f"Base diameter (largest CC, approx): {base_diam}")
    print(f"Base effective diameter (90th pct, sample-BFS): {base_eff_diam:.3f}")

    # 2) Target clustering
    target_C = args.target_clustering if args.target_clustering is not None else base_C
    print(f"Target global clustering: {target_C:.6f}")

    # 3) Run pipeline (uses the earlier functions you have above)
    Gp = rewire_with_hh_and_annealing(
        G_base=G,
        target_clustering=target_C,
        use_spanning_tree=not args.no_tree,
        L=args.L,
        lambda_cn=args.lambda_cn,
        cn_per_node=args.cn_per_node,
        anneal_steps=args.anneal_steps,
        seed=args.seed,
        T0=args.T0,
        Tf=args.Tf,
        tri_alpha=args.tri_alpha,
        tri_beta=args.tri_beta,
    )

    # 4) Report
    new_density = 2 * Gp.number_of_edges() / (Gp.number_of_nodes() * (Gp.number_of_nodes() - 1))
    new_C = nx.transitivity(Gp)
    print(f"Rewired: {Gp.number_of_nodes()} nodes, {Gp.number_of_edges()} edges")
    print(f"New  density = {new_density:.6f} | New global clustering = {new_C:.6f} | Connected? {nx.is_connected(Gp)}")
    print("Degree sequence preserved?", degree_sequence(G) == degree_sequence(Gp))

    # Hard check
    assert_degree_sequence_match(G, Gp)
    print("Degree sequence exactly preserved.")

    new_diam = graph_diameter(Gp, approx_ok=True)
    new_eff_diam = effective_diameter(Gp, q=0.9, k=200)

    print(f"New diameter (largest CC, approx): {new_diam}")
    print(f"New effective diameter (90th pct, sample-BFS): {new_eff_diam:.3f}")

    delta_diam = new_diam - base_diam
    delta_eff = new_eff_diam - base_eff_diam
    print(f"Δ diameter: {delta_diam:+}")
    print(f"Δ effective diameter: {delta_eff:+.3f}")

    # Plots
    tag = "dense_reed98" if args.dense else "sparse_wikivote"
    plot_degree_distribution([G, Gp], ["Base", "Rewired"], filename=f"{tag}_degree_dist")
    plot_degree_cdf(G, Gp, ["Base", "Rewired"], filename=f"{tag}_degree_cdf.png")

    # (Optional) quick visuals — skip for huge graphs or use subgraph sampling inside
    plot_graph(G, title="Base graph", save_path=f"plots_selected_10version/{tag}_base.png", seed=args.seed)
    plot_graph(Gp, title="Rewired graph", save_path=f"plots_selected_10version/{tag}_rewired.png", seed=args.seed)

    # # 5) Save (optional)
    # if args.save_edgelist:
    #     Path(os.path.dirname(args.save_edgelist) or ".").mkdir(parents=True, exist_ok=True)
    #     nx.write_edgelist(Gp, args.save_edgelist, data=False)
    #     print(f"Saved edgelist to: {args.save_edgelist}")


    # if args.save_mtx:
    #     from scipy.sparse import coo_matrix
    #     A = nx.to_scipy_sparse_array(Gp, format="coo")
    #     from scipy.io import mmwrite
    #     Path(os.path.dirname(args.save_mtx) or ".").mkdir(parents=True, exist_ok=True)
    #     mmwrite(args.save_mtx, A)
    #     print(f"Saved Matrix Market to: {args.save_mtx}")


    # python gnn_based_improved.py --sparse --target_clustering 0.1 --tri_alpha 0.5 --tri_beta 0.0 --anneal_steps 100000  --T0 1.5 --Tf 0.005
