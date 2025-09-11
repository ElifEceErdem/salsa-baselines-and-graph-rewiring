#!/bin/bash

# --- Configuration ---
# Get the absolute path of the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Define paths relative to the script's location.
BASE_DIR="$SCRIPT_DIR/../all_datasets/rewired_edge_swap_wiki" # You can change this to your desired base directory
SCRIPT_PATH="$SCRIPT_DIR/synthtetic_networks_split.py"

# Set constant values for validation and test sizes
VAL_SIZE=200
TEST_SIZE=200

# --- Main Script ---

echo "🔍 Starting search in: $BASE_DIR"
echo "🐍 Using script: $SCRIPT_PATH"
echo "---"

# Find all files ending with .edgelist within the base directory and its subfolders
find "$BASE_DIR" -type f -name "*.edgelist" | while read -r edgelist_file; do
    
    echo "Processing $edgelist_file..."
    
    # Execute the Python script with the constant VAL_SIZE and TEST_SIZE
    python "$SCRIPT_PATH" "$edgelist_file" --val_size $VAL_SIZE --test_size $TEST_SIZE

done

echo "---"
echo "All splits completed."