
import argparse
import numpy as np
import yaml
from types import SimpleNamespace
import torch
from tqdm import tqdm
from collections import defaultdict
import pickle
import os
import networkx as nx
import scipy.sparse as sp

from utils.helpers import save_results_incrementally, save_output_network, get_run_date
from utils.helpers import set_random_seed
from utils.helpers import get_np_csr_graph, ppr_for_batch_nodes, calculate_gini 
from utils.evaluator import Evaluator
from models.salsa import RandomWalkSimilarity
from models.random_rec import random_recommendation
from models.local_structural_sim import LocalStructuralSimilarity


def main(args):

    EVAL_RESULTS_PATH = f"{args.output_path}/eval_results_seed{args.seed}/{args.data}"
    OUTPUT_NETWORK_PATH = f"{args.output_path}/output_networks_seed{args.seed}/"

    # Ensure they exist
    os.makedirs(EVAL_RESULTS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_NETWORK_PATH, exist_ok=True)

    # Set random seed
    set_random_seed(args.seed)

    results_path = f"{EVAL_RESULTS_PATH}/{args.model}_eval_results.csv"
    print("[INFO] results_path:", results_path)
    predicted_links = {}
    
    if args.model in ["salsaAuth", "salsaHub"]:

        train_graph_path = args.train_graph_path
        random_walk_similarity = RandomWalkSimilarity(train_graph_path)

        test_edges_path = args.test_edges_path

        # 1. Load test edges in original ID space
        with open(test_edges_path, "rb") as file:
            test_edges_pokec = pickle.load(file)

        # 2. Build ground truth using original IDs
        ground_truth = defaultdict(list)
        for src, dst in test_edges_pokec:
            ground_truth[src].append(dst)

        if args.model.startswith("salsa"):
            output_path = f"{OUTPUT_NETWORK_PATH}_{args.data}_{args.model}_k{args.k}_kcircletrust{args.k_circle_of_trust}_dampingfactor{args.damping_factor}_tol{args.tol}_seed{args.seed}.txt"
            total_nodes = len(test_edges_pokec)

            # Phase 1: Compute circle of trust of the user using Personalized PageRank
            train_undi_graph_path = args.train_undi_graph_path
            num_nodes, ptr_array, nei_array = get_np_csr_graph(train_undi_graph_path)

            config = {}
            
            config['num_walks'] = args.num_walks #1000
            config['walk_length'] = args.walk_length #30
            config['alpha'] =  1-args.damping_factor #by default in the implementation, alpha is restart probability so, we do 1-damping factor
            config['topk'] = args.k_circle_of_trust

            topk_nids_and_ppr_scores_list = np.zeros(shape=(num_nodes, 3, config['topk']), dtype=np.int32)

            print("run ppr...")

            # Gather all unique source nodes from ground_truth
            nodes_to_compute = sorted(set(ground_truth.keys()))
            # Create DataLoader from the specific nodes
            nodes_dl = torch.utils.data.DataLoader(dataset=torch.tensor(nodes_to_compute), batch_size=512)

            # Preallocate results only for those nodes
            topk_nids_and_ppr_scores_list = np.zeros(shape=(len(nodes_to_compute), 3, config['topk']), dtype=np.int32)
            node_index_to_array_index = {nid: i for i, nid in enumerate(nodes_to_compute)}


            idx_start = 0
            for nids in (nodes_dl):
                _N = len(nids)
                topk_nids_and_topk_ppr_scores = ppr_for_batch_nodes(
                    ptr_array=ptr_array, nei_array=nei_array,
                    batch_nodes=nids.numpy(),
                    num_walks=config['num_walks'],
                    walk_length=config['walk_length'],
                    alpha=config['alpha'],
                    topk=config['topk'])
                topk_nids_and_ppr_scores_list[idx_start:idx_start+_N] = topk_nids_and_topk_ppr_scores[:]
                idx_start += _N

            ## NOTE: topk_nids_and_ppr_scores_list.shape = (num_nodes, 3, topk) (3 arrays size of topk)
            ## The first array (index 0 along axis 1) contains the topk most relevant node IDs (neighbors) found via PPR.
            ## The second array (index 1) contains the PPR scores corresponding to these topk neighbors.
            ## The third array (index 2) contains a mask or flag (all ones, indicating valid entries).


            with tqdm(total=total_nodes, desc="Processing nodes") as pbar:
                for i, (node_index, gt_indices) in enumerate(ground_truth.items(), start=1):

                    # Generate recommendations using PageRank
                    k_for_circle_of_trust = args.k_circle_of_trust #500
                    damping_factor = args.damping_factor #0.85
                    k_for_recommendation= args.k

                    tol= args.tol #1e-8

                    if args.model == "salsaAuth":
                        array_index = node_index_to_array_index[node_index]
                        recommendations = random_walk_similarity.twitter_wtf_new_version(
                            topk_nids_and_ppr_scores_list, 
                            array_index, 
                            k_for_circle_of_trust, tol, damping_factor, k_for_recommendation, salsa_auth=True
                        )
                    
                    elif args.model == "salsaHub":
                        array_index = node_index_to_array_index[node_index]
                        recommendations = random_walk_similarity.twitter_wtf_new_version(
                            topk_nids_and_ppr_scores_list, 
                            array_index, 
                            k_for_circle_of_trust, tol, damping_factor, k_for_recommendation, salsa_auth=False
                        )

                    # before 
                    predicted_links[node_index] = [x[1] for x in recommendations]

                    # Update the progress bar every 10,000 nodes
                    if i % 1000 == 0:
                        pbar.update(1000)
                
                # Update the progress bar for any remaining nodes
                remaining_nodes = total_nodes % 1000
                if remaining_nodes != 0:
                    pbar.update(remaining_nodes)

    elif args.model == "common_neighbors" or args.model == "random":
        output_path = f"{OUTPUT_NETWORK_PATH}_{args.data}_{args.model}_seed{args.seed}.txt"

        # 1. Load test edges in original ID space
        with open(args.test_edges_path, "rb") as file:
            test_edges_pokec = pickle.load(file)
        ground_truth = defaultdict(list)
        for src, dst in test_edges_pokec:
            ground_truth[src].append(dst)
        # 2. Load graph as adjacency dict/list for these methods
        train_undi_graph_path = args.train_undi_graph_path
        # It should return (num_nodes, ptr_array, nei_array), but we want a dict/list
        # You'll need a helper to convert np_csr.pkl to dict/list of neighbors:
        num_nodes, ptr_array, nei_array = get_np_csr_graph(train_undi_graph_path)
        adj_dict = {}
        for u in range(num_nodes):
            ptr, degree = ptr_array[u]
            neighbors = nei_array[ptr:ptr+degree]
            adj_dict[u] = list(neighbors)

        if args.model == "common_neighbors":
            lss = LocalStructuralSimilarity(adj_dict, {}, {})
            users = set([src for src, _ in test_edges_pokec])
            with tqdm(total=len(users), desc="Common Neighbor") as pbar:
                for u in users:
                    scores = []
                    exclude = set(adj_dict.get(u, []))
                    exclude.add(u)
                    for v in range(num_nodes):
                        if v not in exclude:
                            score = lss.common_neighbors(u, v)
                            scores.append((v, score))
                    topk = sorted(scores, key=lambda x: x[1], reverse=True)[:args.k]
                    predicted_links[u] = [v for v, _ in topk]
                    pbar.update(1)
        elif args.model == "random":
            users = set([src for src, _ in test_edges_pokec])
            # You can use the provided function
            predicted_links = random_recommendation(
                np.array(test_edges_pokec), adj_dict, args.k, seed=args.seed
            )



    predicted_edges = [(int(u), int(v)) for u, links in predicted_links.items() for v in links]
    # Save the edges to an output network file
    save_output_network(predicted_edges, output_path=output_path)

    # ---- Compute fairness & clustering metrics on predicted network ----
    rows = [u for u, v in predicted_edges]
    cols = [v for u, v in predicted_edges]
    if rows and cols:
        N = max(max(rows), max(cols)) + 1
        adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        degree_values = np.array(adj.sum(axis=0)).flatten()  
        gini_index_degree = calculate_gini(degree_values)
        one_minus_gini = 1 - gini_index_degree
        global_clustering = nx.average_clustering(nx.from_scipy_sparse_array(adj))
    else:
        one_minus_gini = np.nan
        global_clustering = np.nan

    # Compute AUC if available
    auc = 0  # Placeholder for AUC calculation (update if AUC is computed)

    # Prepare evaluation results
    run_date = get_run_date()

    evaluator = Evaluator(k=args.k)

    # Evaluate predictions
    evaluation_results = evaluator.evaluate_all_predictions(predicted_links, test_edges_pokec, args.k)

    # Extract metrics
    mean_precision = evaluation_results["mean_precision"]
    mean_recall = evaluation_results["mean_recall"]
    mean_ndcg = evaluation_results["mean_ndcg"]
    map_at_k = evaluation_results["map_at_k"]
    auc = evaluation_results["auc"]
    #map_score = evaluation_results["map"]

    eval_results = {
        "model": args.model,
        "precision": mean_precision,
        "recall": mean_recall,
        "ndcg": mean_ndcg,
        "map_at_k": map_at_k,
        "auc": auc,
        "1-gini index": one_minus_gini,
        "Global clustering": global_clustering,
        "k": args.k,
        "k_circle_of_trust": args.k_circle_of_trust,
        "damping_factor": args.damping_factor,
        "tol":args.tol,
        "seed": args.seed,
        "run_date": run_date
    }

    save_results_incrementally(results_path, eval_results)
    print("Results saved successfully.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run SALSA training and evaluation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--train_graph_path", type=str, help="Path to the training graph")
    parser.add_argument("--train_undi_graph_path", type=str, help="Path to the undirected training graph")
    parser.add_argument("--test_edges_path", type=str, help="Path to the test edges")
    parser.add_argument("--data", type=str, help="Dataset to use")
    parser.add_argument("--model", type=str, choices=["salsaAuth", "salsaHub", "common_neighbors", "random"], help="Model to use")
    parser.add_argument("--k", type=int, help="Number of top predictions for evaluation")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--k_circle_of_trust", type=int, help="Number of people in the circle of trust")
    parser.add_argument("--damping_factor", type=float, help="Damping factor for personalization")
    parser.add_argument("--tol", type=float, help="Tolerance for convergence")
    parser.add_argument("--num_walks", type=int, help="Number of walks per node for PPR computation")
    parser.add_argument("--walk_length", type=int, help="Walk length for PPR computation")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility")
    parser.add_argument("--output_path", type=int, help="Path to save output network and results")

    # Parse CLI args (but do not call main yet!)
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Merge: CLI > YAML
    cli_args = vars(args)
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    
    args = SimpleNamespace(**config)

    main(args)



