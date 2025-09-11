#!/bin/bash

# --- Check for Command-Line Argument ---
if [ "$#" -ne 1 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: $0 /path/to/dataset_directory"
    exit 1
fi

# --- Configuration ---
DATASET_BASE_PATH="$1"
DATASET_NAME=$(basename "$DATASET_BASE_PATH")
MODELS=("salsaAuth" "common_neighbors" "random")
SEEDS=(7 42 123 1000 2025 2307 3407 4508 5510 6600)
EXTRA_ARGS="--config config.yaml"

# --- Dynamic Path Generation ---
train_graph_path="${DATASET_BASE_PATH}/train_graph.pkl"
train_undi_graph_path="${DATASET_BASE_PATH}/train_undi_graph.np_csr.pkl"
test_edges_path="${DATASET_BASE_PATH}/eval-recall/recall_test_pos_edgelist.pkl"

# --- Main Execution Loop ---
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "➡️  Running model=$model, data=$DATASET_NAME, seed=$seed"
    python main.py $EXTRA_ARGS \
      --train_graph_path "$train_graph_path" \
      --train_undi_graph_path "$train_undi_graph_path" \
      --test_edges_path "$test_edges_path" \
      --model "$model" \
      --data "$DATASET_NAME" \
      --seed "$seed" \
  done
done

echo "All tasks completed."