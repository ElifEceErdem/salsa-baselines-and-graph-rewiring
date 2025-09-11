#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BASE_DIR="$SCRIPT_DIR/../all_datasets/rewired_edge_swap_wiki" # you can change this to your desired base directory
SCRIPT_PATH="$SCRIPT_DIR/ssnet_salsa_dataprep.py"

# for d in "$BASE_DIR"/network*; do
for d in "$BASE_DIR"/ws*; do
    if [ -d "$d" ]; then
        echo "Processing $d"
        python "$SCRIPT_PATH" "$d"
    fi
done

echo "All folders processed."
