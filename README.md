# When can Social Network Recommendations Improve Precision and Mitigate Inequality? — Code & Data

> Reproducible code for the paper **“When can Social Network Recommendations Improve Precision and Mitigate Inequality?”**. This repo provides: (i) graph rewiring to control structure (triangle‑focused edge‑swap and a GNN‑based), (ii) baselines for link recommendation (Random, Common Neighbors/Local Structural Similarity, SALSA, plus data preperation for LightGCN, PPRGo, SSNet, Tail‑STEAK), and (iii) an evaluation pipeline reporting Precision\@10, degree fairness.

---

## Table of Contents

* [What’s here](#whats-here)
* [Repo layout](#repo-layout)
* [Installation](#installation)
* [Data & formats](#data--formats)
* [Data availability & download](#data-availability--download)
* [Run baseline recommender](#run-baseline-recommender)
* [Configuration](#configuration)
* [Rewiring methods](#rewiring-methods)
* [Reproducing paper results](#reproducing-paper-results)

---

## What’s here

* **Baselines**: This repo includes simple recommenders like Random, Common Neighbors, and SALSA. It also includes helper code to easily run more advanced models from the paper, such as LightGCN, PPRGo, [SSNet](https://github.com/CGCL-codes/ssnet/tree/master), and [Tail‑STEAK](https://github.com/antman9914/Tail-STEAK/tree/main)
* **Graph rewiring**:
  * `edge_rewiring/edge_swap.py` — triangle‑focused **degree‑preserving** swaps to systematically **decrease clustering**.
  * `edge_rewiring/gnn_based.py` — a **GNN‑driven** scoring + constrained projection to **target clustering** while preserving **density** and **degree distribution** (via Wasserstein/degree caps).
  * `edge_rewiring/gnn_based_improved.py` — **(New)** An improved hybrid approach. Uses **SGC encoding** for structural scoring, **Weighted Havel-Hakimi** (or Min-Cost Flow) to enforce the **exact degree sequence**, and **Simulated Annealing** to precisely tune **clustering**.
  * `edge_rewiring/batch_graph_generation.py` — Automation script to generate large batches of rewired graphs across varying clustering targets.
* **Evaluation**: unified metrics (P\@10, 1−Gini), CSV outputs, plotting hooks.

> **Notes**: All graphs are treated as **undirected** with **zero‑indexed node IDs**. We average over multiple seeds (10 on empirical; 2 on synthetic) as in the paper.

---

## Repo layout

```
.
├─ edge_rewiring/
│  ├─ batch_graph_generation.py
|  ├─ edge_swap.py
|  ├─ gnn_based.py
│  └─ gnn_based_improved.py
├─ plots_generation/
│  ├─ fairness_tail_examination.py
│  ├─ precision_fairness_tradeoff.py
│  ├─ synthetic_net_comparison.py
├─ salsa/
│  ├─ datasets/
│  │  ├─ all_datasets/
│  │  ├─ data_prep/
│  ├─ models/
│  │  ├─ local_structural_sim.py
│  │  ├─ random_rec.py
│  │  └─ salsa.py
│  ├─ outputs/
│  ├─ utils/
│  │  ├─ evaluator.py
│  │  └─helpers.py
│  ├─ config.yaml
│  ├─ main.py
│  ├─ reed98_exp.sh
│  └─ rewired_gnn_synthetic_exp.sh
├─ environment.yml
└─ LICENSE
└─ README.md

```

---

## Installation

We provide a fully pinned Conda spec for exact reproducibility and a minimal setup for portability.

```bash
conda env create -f environment.yml
# (optional) override the env name
# conda env create -f environment.yml -n salsa_and_edge_rewire

conda activate salsa_and_edge_rewire # or the name defined in the YAML
```

---

## Data & formats

* **Edgelists**: text files with `u v` per line, whitespace‑separated, **undirected**, **zero‑indexed**.
* Place empirical graphs under `salsa/datasets/{twitter,pokec,wiki,reed98}`.
* Fully synthetic graphs live under `ws_graphs/` (Watts–Strogatz) and `pl_graphs/` (Power‑law/cluster).
* Semi‑synthetic graphs (edge‑swap and GNN‑rewired) go under the corresponding `rewired_*` folders.

Example (first 4 lines):

```
0 12
0 91
1 37
1 42
```

---

## Data availability & download

The full datasets are large, so we host them externally.

- **Google Drive**: [All Processed Datasets]()


## Run baseline recommender

Use `main.py` (either via CLI flags or the YAML config). Example CLI:

```bash
python main.py \
  --model salsa \
  --dataset wiki \
  --data_dir salsa/datasets/wiki \
  --topk 100 \
  --alpha 0.7 \
  --p_at_k 10 \
  --out_dir salsa/outputs/wiki_salsa_topk100
```

Or with a YAML config (see below),

```bash
python main.py --config config.yaml
```

Results (CSV) are written under `salsa/outputs/...`.

---

## Rewiring methods

### Improved Degree-Preserving Generator (SGC + Annealing)

A hybrid method that combines representation learning with combinatorial optimization to **target global clustering** while maintaining the **exact degree sequence** and **connectivity** of the base graph.

**Pipeline:**
1. **Encoder**: Computes structural node embeddings using **Simple Graph Convolution (SGC)** (with optional $A^2$ terms for triangle awareness).
2. **Scoring**: Ranks potential edges using Centered Cosine Similarity or SGC-Tri scoring.
3. **Projection**: Uses **Weighted Havel-Hakimi** (fallback to Min-Cost Flow) to select edges that strictly match the original degree sequence.
4. **Annealing**: Refines the graph using **degree-preserving edge swaps** to reach the exact target clustering coefficient.

**Example**:

```bash
python edge_rewiring/gnn_based_improved.py \
  --sparse \
  --target_clustering 0.1 \
  --anneal_steps 100000 \
  --T0 1.5 \
  --Tf 0.005 \
  --seed 7
```

To generate comprehensive datasets (e.g., sweeping clustering from 0.0 to 0.6) for experiments, use the batch script. This automates the improved generator across multiple seeds and targets.

```bash
python edge_rewiring/batch_graph_generation.py
```
  
  
## Reproducing paper results

1. **Prepare graphs**
   Place empirical graphs under `salsa/datasets/{twitter,pokec,wiki,reed98}`. Generate WS/PL graphs under `ws_graphs/` and `pl_graphs/`.
2. **(Optional) Rewire**
   Use `edge_swap.py` to **decrease** clustering or `gnn_based.py` to target **higher/lower** clustering while keeping density & degree distribution stable. Write outputs to `rewired_*` folders.
3. **Run recommenders**
   Execute `main.py` or the provided shell scripts (e.g., `./reed98_exp.sh`, `./rewired_gnn_synthetic_exp.sh`). SALSA is implemented here; other GNN baselines use their official codebases.
4. **Collect results into CSVs**
   The plotting scripts expect aggregated CSVs named like the ones in `plots_generation/` (e.g., `salsa_seed_results.csv`, `ssnet_seed_results.csv`, `tailsteak_seed_results.csv`, plus synthetic variants such as `ws_pl_all_results.csv` and `edge_swap_all_results.csv.csv`). Keep the same column names used in the paper: `dataset`, `model`/`Model`, `p10`, `1-gini` (or `fairness`), `Global clustering`, etc.
5. **Generate figures**
   From the repo root:
   **Pareto front (precision vs fairness) across models**
   ```bash
      python plot_generation/precision_fairness_tradeoff.py \
      --salsa-csv "/path/to/salsa_seed_results.csv" \
      --ssnet-csv "/path/to/ssnet_seed_results.csv" \
      --tailsteak-csv "/path/to/tailsteak_seed_results.csv" \
      --cn-csv "/path/to/common_neighbor_seed_results.csv" \
      --random-csv "/path/to/random_seed_results.csv" \
      --output-dir "/path/to/output_folder"
   ```

   **Synthetic: results for WS/PL families**
   ```bash
      python plot_generation/synthetic_net_comparison.py ws_pl \
      --salsa-csv "/path/to/salsa_all_results.csv" \
      --ssnet-csv "/path/to/ssnet_all_results.csv" \
      --tailsteak-csv "/path/to/tailsteak_all_results.csv" \
      --props-csv "/path/to/networks_info.csv" \
      --output-dir "/path/to/plots_folder"
   ```
   **Synthetic: results for degree‑preserving edge‑swap rewiring**
   ```bash
      python plot_generation/synthetic_net_comparison.py edge_swap
      --salsa-csv "/path/to/salsa_all_results.csv" \
      --ssnet-csv "/path/to/ssnet_all_results.csv" \
      --tailsteak-csv "/path/to/tailsteak_all_results.csv" \
      --props-csv "/path/to/networks_info.csv" \
      --output-dir "/path/to/plots_folder"
   ```
   
   **Synthetic: results for GNN‑based rewiring**
   ```bash
      python plot_generation/synthetic_net_comparison.py gnn \
   --salsa-csv "/path/to/salsa_all_results.csv" \
   --ssnet-csv "/path/to/ssnet_all_results.csv" \
   --tailsteak-csv "/path/to/tailsteak_all_results.csv" \
   --props-csv "/path/to/networks_info.csv" \
   --output-dir "/path/to/plots_folder"
   ```

   ***Fairness tail examination (long‑tail analysis)**
   ```bash
      python plot_generation/tailsteak_fairness_tail_examination.py \
      --original-graph "/path/to/socfb-Reed98.edgelist" \
      --pprgo-predictions "/path/to/pprgo_predicted_network_top10.npy" \
      --tailsteak-predictions "/path/to/tailsteak_top_10_predicted_edges_seed_7.txt" \
      --output-dir "/path/to/output_folder" \
      --output-filename "my_comparison_plot.png" \
      --num-bins 15
   ```
---


