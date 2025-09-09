# A Structural Dilemma: The Trade‑off Between Precision and Fairness in Link Recommendations — Code & Data

> Reproducible code for the paper **“A Structural Dilemma: The Trade‑off Between Precision and Fairness in Link Recommendations”**. This repo provides: (i) graph rewiring to control structure (triangle‑focused edge‑swap and a GNN‑based), (ii) baselines for link recommendation (Random, Common Neighbors/Local Structural Similarity, SALSA, plus data preperation for LightGCN, PPRGo, SSNet, Tail‑STEAK), and (iii) an evaluation pipeline reporting Precision\@10, 1−Gini (degree fairness).

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

* **Baselines**: Random recommender, Common Neighbors/Local Structural Similarity, and **SALSA** (our re‑implementation). Hooks provided to run LightGCN, PPRGo, SSNet, and Tail‑STEAK as used in the paper.
* **Graph rewiring**:

  * `edge_rewiring/edge_swap.py` — triangle‑focused **degree‑preserving** swaps to systematically **decrease clustering**.
  * `edge_rewiring/gnn_based.py` — a **GNN‑driven** scoring + constrained projection to **target clustering** while preserving **density** and **degree distribution** (via Wasserstein/degree caps) and encouraging connectivity (MST backbone).
* **Datasets**: empirical (Twitter‑9K, Pokec, Wiki, Reed98), fully synthetic (Watts–Strogatz, Power‑law/cluster), and semi‑synthetic (edge‑swap and GNN‑rewired variants).
* **Evaluation**: unified metrics (P\@10, 1−Gini), CSV outputs, plotting hooks.

> **Notes**: All graphs are treated as **undirected** with **zero‑indexed node IDs**. We average over multiple seeds (10 on empirical; 2 on synthetic) as in the paper.

---

## Repo layout

```
.
├─ edge_rewiring/
│  ├─ edge_swap.py
│  └─ gnn_based.py
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

The full datasets are large (≈29 GB compressed), so we host them externally.

- **Google Drive**: [All Processed Datasets](https://drive.google.com/drive/folders/1i-FVCXLoVCtrCz6j_JQP7WOg6jXOUjol?usp=sharing)


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

### Edge‑swap (degree‑preserving)

Fast swaps that preserve the degree sequence and typically **decrease** clustering. Useful for generating lower‑clustering controls.

**Example**:

```bash
python edge_rewiring/edge_swap.py \
  --input salsa/datasets/wiki/graph.edgelist \
  --output salsa/datasets/rewired_edge_swap_wiki/wiki_swap.edgelist \
  --num_swaps 200000 \
  --avoid_self_loops True \
  --avoid_multi_edges True
```

### GNN‑based optimizer (target clustering)

Learns edge scores via a GNN (2‑layer GCN + dot‑product decoder) and **projects** to an adjacency under hard constraints:

* preserve **density** (±ε)
* preserve **degree distribution** (Wasserstein + degree caps; KL monitoring)
* **target global clustering** (projection enforces an upper bound)
* encourage **connectivity** via an MST backbone

**Example**:

```bash
python edge_rewiring/gnn_based.py \
  --wass_weight 40 \
  --alpha 0.3 \
  --temperature 0.05 \
  --early_stop \
  --mst_keep_ratio 1 \
  --sparse \
  --num_layers 2 \
  --target_clustering 0.05 \
  --output salsa/datasets/rewired_gnn_graphs/wiki_gnn_rewired.edgelist
  --seed 7"
```

Implementation details as in the paper: GCN hidden dim 32, Adam lr 0.01; key knobs alpha ∈ {0.05,0.1,0.3}, MST ratio ∈ {0.25,0.5,0.75,1}.

---

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


