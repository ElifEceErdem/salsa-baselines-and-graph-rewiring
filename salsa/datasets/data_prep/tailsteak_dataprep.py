import argparse
import json
import random
from collections import defaultdict
from tqdm import tqdm
import os

# === CLI Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--edge_list_path', type=str, required=True, help='Path to input edgelist')
parser.add_argument('--output_path', type=str, required=True, help='Directory to save graph.txt')
parser.add_argument('--val_ratio', type=float, default=0.08)
parser.add_argument('--test_ratio', type=float, default=0.91)
parser.add_argument('--val_neg_k', type=int, default=9)
parser.add_argument('--test_neg_k', type=int, default=9)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# === Config ===
INPUT_PATH = args.edge_list_path
OUTPUT_DIR = args.output_path
VAL_RATIO = args.val_ratio
TEST_RATIO = args.test_ratio
VAL_NEG_K = args.val_neg_k
TEST_NEG_K = args.test_neg_k
SEED = args.seed
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'graph.txt')

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Helper: load undirected edges without duplicates ===
def load_edges(path):
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u < v:
                yield u, v

# === Phase 1: Build neighbors and split edges to temp files ===
train_tmp = os.path.join(OUTPUT_DIR, 'train.tmp')
val_tmp = os.path.join(OUTPUT_DIR, 'val.tmp')
test_tmp = os.path.join(OUTPUT_DIR, 'test.tmp')
neighbors = defaultdict(set)

with open(train_tmp, 'w') as t_train, open(val_tmp, 'w') as t_val, open(test_tmp, 'w') as t_test:
    for u, v in tqdm(load_edges(INPUT_PATH), desc='Splitting edges'):
        r = random.random()
        if r < TEST_RATIO:
            t_test.write(f"{u} {v}\n")
        elif r < TEST_RATIO + VAL_RATIO:
            t_val.write(f"{u} {v}\n")
        else:
            neighbors[u].add(v)
            neighbors[v].add(u)
            t_train.write(f"{u} {v}\n")

# In-memory list of all node IDs
all_nodes = list(neighbors.keys())

# === Negative sampler using only neighbors and node list ===
def sample_negatives(u, v, k):
    invalid = neighbors[u] | {u, v}
    negs = []
    while len(negs) < k:
        cand = random.choice(all_nodes)
        if cand not in invalid:
            negs.append(cand)
    return negs

# === Phase 2: Stream JSON structure ===
with open(OUTPUT_JSON, 'w') as out:
    # Start JSON object
    out.write('{"base_graph":[[')
    
    first = True
    with open(train_tmp) as fin:
        for line in tqdm(fin, desc='Writing base_graph'):
            u, v = map(int, line.split())
            if not first:
                out.write(',')
            out.write(f'[{u},{v}]')
            first = False
    out.write(']],"val_set":[')
    # val_set entries
    first = True
    with open(val_tmp) as fin:
        for line in tqdm(fin, desc='Writing val_set'):
            u, v = map(int, line.split())
            negs = sample_negatives(u, v, VAL_NEG_K)
            entry = [u, v] + negs
            random.shuffle(entry[1:])
            if not first:
                out.write(',')
            out.write(json.dumps(entry))
            first = False
    out.write('],"test_set":[')
    # test_set entries
    first = True
    with open(test_tmp) as fin:
        for line in tqdm(fin, desc='Writing test_set'):
            u, v = map(int, line.split())
            negs = sample_negatives(u, v, TEST_NEG_K)
            entry = [u, v] + negs
            random.shuffle(entry[1:])
            if not first:
                out.write(',')
            out.write(json.dumps(entry))
            first = False
    
    out.write(']}')

print(f"Saved graph_data-format JSON to {OUTPUT_JSON}")

