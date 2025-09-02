#!/bin/bash

models=("salsaAuth" "common_neighbors" "random")
datasets=("reed98")
seeds=(7 42 123 1000 2025 2307 3407 4508 5510 6600)

# Add any additional args you want to pass on CLI. (otherwise read from config.yaml)
EXTRA_ARGS="--config config.yaml"

train_graph_path="/var/scratch/eerdem/final_saved_graphs/reed98/train_graph.pkl"
train_undi_graph_path="/var/scratch/eerdem/final_saved_graphs/reed98/train_undi_graph.np_csr.pkl"
test_edges_path="/var/scratch/eerdem/final_saved_graphs/reed98/eval-recall/recall_test_pos_edgelist.pkl"

for model in "${models[@]}"; do
  for data in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running model=$model, data=$data, seed=$seed"
      python main.py $EXTRA_ARGS --train_graph_path $train_graph_path \
      --train_undi_graph_path $train_undi_graph_path \
      --test_edges_path $test_edges_path \
      --model $model --data $data --seed $seed
    done
  done
done