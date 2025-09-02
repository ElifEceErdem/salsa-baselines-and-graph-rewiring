import random
import torch
import numpy as np
import os.path as osp
import pickle as pkl
import gc

import pandas as pd
import os
import networkx as nx
from datetime import datetime
import numba
from tqdm import tqdm

def set_random_seed(seed):
    """
    Set the random seed for reproducibility across different libraries.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pickle(filename):
    with open(filename, "rb") as f:
        gc.disable()
        obj = pkl.load(f)
        gc.enable()
    return obj

def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        pkl.dump(obj, f)

def save_results_incrementally(results_path, new_results):
    """
    Save results incrementally to a CSV file.

    Args:
        results_path (str): Path to the results file.
        new_results (dict or pd.DataFrame): New results to append to the file.
    """
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
    else:
        existing_results = pd.DataFrame()

    # Ensure new_results is a DataFrame
    if isinstance(new_results, dict):
        new_results = pd.DataFrame([new_results])  # Convert dict to DataFrame (single row)

    updated_results = pd.concat([existing_results, new_results], ignore_index=True)
    updated_results.to_csv(results_path, index=False)
    print(f"Results updated and saved to {results_path}")

def save_output_network(predicted_edges, output_path="output_network.edgelist"):
    
    output_graph = nx.DiGraph()
    output_graph.add_edges_from(predicted_edges)
    nx.write_edgelist(output_graph, output_path, data=False)
    print(f"Output network saved to {output_path}")

def get_run_date():
    """
    Get the current date and time as a string.

    Returns:
        str: Current date and time in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def get_np_csr_graph(csr_file):
    if osp.exists(csr_file):
        print("load csr graph from cached file")
        csr_graph = load_pickle(csr_file)
        ptr_array = csr_graph['ptr_array']    # prt_array: shape=(num_nodes, 2), [[pointer, degree], [pointer, degree], ... ]
        nei_array = csr_graph['nei_array']    # nei_array: shape=(num_edges,)
        num_nodes = len(ptr_array)
    else:
        print("convert dgl graph to csr graph")
        if dgl_graph_file is not None:
            dgl_g = load_pickle(osp.join(data_root, dgl_graph_file))
        else:
            dgl_g = load_pickle(osp.join(data_root, "train_graph.pkl"))
        num_nodes = dgl_g.num_nodes()
        num_edges = dgl_g.num_edges()
        
        ptr_array = np.empty(shape=(num_nodes, 2), dtype=np.int32)
        nei_array = np.empty(shape=(num_edges,), dtype=np.int32)
        
        ptr_array[:, 1] = dgl_g.out_degrees(dgl_g.nodes()).numpy()
        ptr = 0
        for nid in tqdm(range(num_nodes)):
            ptr_array[nid, 0] = ptr
            degree = ptr_array[nid, 1]
            nei_array[ptr : ptr + degree] = dgl_g.successors(nid).numpy()
            ptr += degree
        
        csr_graph = {'ptr_array': ptr_array, 'nei_array': nei_array}
        save_pickle(csr_file, csr_graph)
    
    return num_nodes, ptr_array, nei_array

@numba.jit(nopython=True)
def ppr_for_one_node(ptr_array, nei_array, source, num_walks, walk_length, alpha, topk):
    res = numba.typed.Dict.empty(
        key_type=numba.types.int32,
        value_type=numba.types.int32,
    )
    res[source] = 1
    _, degree = ptr_array[source]
    if degree <= 0:
        topk_nids = np.array([[source],[1]], dtype=np.int32)
        return topk_nids
    for _ in range(num_walks):
        u = source
        for _ in range(walk_length):
            ptr, degree = ptr_array[u]
            p = random.random()
            if p > alpha or degree <= 0:
                u = source
                continue
            offset = np.random.randint(low=0, high=degree)
            v = nei_array[ptr + offset]
            if v in res:
                res[v] += 1
            else:
                res[v] = 1
            u = v
    ppr_nids = np.array(list(res.keys()), dtype=np.int32)
    ppr_scores = np.array(list(res.values()), dtype=np.int32)
    
    ind = np.argsort(ppr_scores)[-topk:][::-1]
    topk_nids = ppr_nids[ind]
    topk_ppr_scores = ppr_scores[ind]
    
    re = np.empty(shape=(2, len(topk_nids)), dtype=np.int32)
    re[0] = topk_nids
    re[1] = topk_ppr_scores
    return re

@numba.jit(nopython=True, parallel=True)
def ppr_for_batch_nodes(ptr_array, nei_array, batch_nodes, num_walks, walk_length, alpha, topk, punish=0):
    num_nodes = len(batch_nodes)
    # topk_nids_and_ppr_scores_list = [np.zeros(shape=(2,1), dtype=np.int32)] * num_nodes
    topk_nids_and_ppr_scores_list = np.zeros(shape=(num_nodes, 3, topk), dtype=np.int32)
    for i in numba.prange(num_nodes):
        nid = batch_nodes[i]
        topk_nids_and_ppr_scores = ppr_for_one_node(
            ptr_array, nei_array, nid, num_walks, walk_length, alpha, topk
        )
        _N = len(topk_nids_and_ppr_scores[0])
        topk_nids_and_ppr_scores_list[i, :2, :_N] = topk_nids_and_ppr_scores[:, :_N]  # do not use list.append !
        topk_nids_and_ppr_scores_list[i, 2, :_N] = 1
    return topk_nids_and_ppr_scores_list


# Helper to compute Gini
def calculate_gini(values):
    values = np.array(values).flatten()
    if np.amin(values) < 0:
        values -= np.amin(values)
    if np.sum(values) == 0:
        return 0.0
    values = np.sort(values)
    n = values.shape[0]
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
    return gini