import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load SALSA results
salsa = pd.read_csv("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/fig1/salsa_seed_results.csv").rename(columns={
    "network": "dataset",
    "1-gini index": "fairness",
    "precision": "p10"
})
salsa["Model"] = "SALSA"

# Load SSNet results
ss = pd.read_csv("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/fig1/ssnet_seed_results.csv").rename(columns={
    "Network": "dataset",
    "1-gini": "fairness",
    "p10": "p10",
    "Model": "base_model",
    "Seed": "seed"
})
ss["Model"] = ss.apply(lambda r: r["base_model"] if r["Mode"].lower() == "normal" else f"{r['base_model']}+SSNet", axis=1)

# Load Tail-STEAK results
tail = pd.read_csv("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/fig1/tailsteak_seed_results.csv").rename(columns={
    "p@10": "p10",
    "1_minus_gini": "fairness"
})
tail["Model"] = "Tail-STEAK"

# Load Common Neighbors results
# This part assumes the CSV might have 'network' or 'dataset' and handles it.
cn_df = pd.read_csv("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/fig1/common_neighbor_seed_results.csv")
cn = cn_df.rename(columns={
    "network": "dataset",
    "1-gini index": "fairness",
    "precision": "p10"
})
cn["Model"] = "Common Neighbors"


# Load Random Recommender results
# This part assumes the CSV might have 'network' or 'dataset' and handles it.
rand_df = pd.read_csv("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/fig1/random_seed_results.csv")
rand = rand_df.rename(columns={
    "network": "dataset",
    "1-gini index": "fairness",
    "precision": "p10"
})
rand["Model"] = "Random Recommender"


# Combine data
df = pd.concat([
    salsa[["dataset", "seed", "Model", "fairness", "p10"]],
    ss[["dataset", "seed", "Model", "fairness", "p10"]],
    tail[["dataset", "seed", "Model", "fairness", "p10"]],
    cn[["dataset", "seed", "Model", "fairness", "p10"]],
    rand[["dataset", "seed", "Model", "fairness", "p10"]]
], ignore_index=True)


# --- Plotting Section ---
# Create a 1x4 grid of subplots that share x and y axes.
fig, axes = plt.subplots(1, 4, figsize=(34, 7), sharex=True, sharey=True)
sns.set(style="whitegrid", font_scale=2)

marker_list = ["o", "s", "D", "^", "v", "P", "X", "*"]
color_palette = sns.color_palette("tab10", len(marker_list))

# Loop through each subplot and its corresponding dataset name.
# We use enumerate to get the index `idx` of the current subplot.
for idx, (ax, ds) in enumerate(zip(axes, ["twitter", "pokec", "wiki", "reed98"])):
    sub = df[df["dataset"] == ds]
    models = sub["Model"].unique()

    for i, (model, marker) in enumerate(zip(models, marker_list)):
        model_data = sub[sub["Model"] == model]

        # Compute mean and standard deviation for error bars
        fair_mean = model_data["fairness"].mean()
        p10_mean = model_data["p10"].mean()
        fair_std = model_data["fairness"].std()
        p10_std = model_data["p10"].std()

        # Plot mean points with error bars representing ± standard deviation
        ax.errorbar(fair_mean, p10_mean,
                    xerr=fair_std,
                    yerr=p10_std,
                    fmt=marker, markersize=14, capsize=8, capthick=2,
                    label=model, color=color_palette[i], mec='black', mew=0.5)

    # Set subplot titles and labels
    ax.set_title(ds.capitalize(), fontsize=30)
    ax.set_xlabel("Fairness (1 – Gini)", fontsize=30)
    
    # --- KEY CHANGE HERE ---
    # Only set the y-axis label for the very first plot (index 0).
    if idx == 0:
        ax.set_ylabel("Precision @ 10", fontsize=30)
    
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(loc="best", fontsize=19.5)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/fig1/improved_fairness_precision_variance_by_model_y_axis_once.pdf", dpi=300)