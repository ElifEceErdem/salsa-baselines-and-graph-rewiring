#!/bin/bash

models=("salsaAuth")
seeds=(7 42)

EXTRA_ARGS="--config config.yaml"

BASE_PATH="./datasets/rewired_gnn_graphs"

for net_id in {1..8}; do
  for version in {1..10}; do
    network="gnn_synthetic_network${net_id}_v${version}"
    network_dir="${BASE_PATH}/${network}"

    # Check if the directory exists before proceeding
    if [ ! -d "$network_dir" ]; then
        echo "⚠️  Warning: Directory not found, skipping: $network_dir"
        continue
    fi

    train_graph_path="${network_dir}/train_graph.pkl"
    train_undi_graph_path="${network_dir}/train_undi_graph.np_csr.pkl"
    test_edges_path="${network_dir}/eval-recall/recall_test_pos_edgelist.pkl"

    for model in "${models[@]}"; do
      for seed in "${seeds[@]}"; do
        echo "Running model=$model, network=$network, seed=$seed"
        python main.py $EXTRA_ARGS \
          --train_graph_path "$train_graph_path" \
          --train_undi_graph_path "$train_undi_graph_path" \
          --test_edges_path "$test_edges_path" \
          --model "$model" \
          --data "$network" \
          --seed "$seed"
      done
    done
  done
done