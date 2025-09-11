import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_arguments():
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a fairness vs. precision plot from model result CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--salsa-csv", required=True, help="Path to SALSA results CSV.")
    parser.add_argument("--ssnet-csv", required=True, help="Path to SSNet results CSV.")
    parser.add_argument("--tailsteak-csv", required=True, help="Path to Tail-STEAK results CSV.")
    parser.add_argument("--cn-csv", required=True, help="Path to Common Neighbors results CSV.")
    parser.add_argument("--random-csv", required=True, help="Path to Random Recommender results CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output plot.")
    parser.add_argument(
        "--output-filename",
        default="fairness_precision_comparison.pdf",
        help="Filename for the output PDF plot."
    )
    return parser.parse_args()

def load_and_process_data(args):
    """Loads all CSVs, processes them, and returns a single DataFrame."""
    print("Loading and processing data...")
    
    try:
        salsa = pd.read_csv(args.salsa_csv).rename(columns={
            "network": "dataset", "1-gini index": "fairness", "precision": "p10"
        })
        salsa["Model"] = "SALSA"

        ss = pd.read_csv(args.ssnet_csv).rename(columns={
            "Network": "dataset", "1-gini": "fairness", "p10": "p10", "Model": "base_model", "Seed": "seed"
        })
        ss["Model"] = ss.apply(lambda r: r["base_model"] if r["Mode"].lower() == "normal" else f"{r['base_model']}+SSNet", axis=1)

        tail = pd.read_csv(args.tailsteak_csv).rename(columns={"p@10": "p10", "1_minus_gini": "fairness"})
        tail["Model"] = "Tail-STEAK"

        cn = pd.read_csv(args.cn_csv).rename(columns={
            "network": "dataset", "1-gini index": "fairness", "precision": "p10"
        })
        cn["Model"] = "Common Neighbors"

        rand = pd.read_csv(args.random_csv).rename(columns={
            "network": "dataset", "1-gini index": "fairness", "precision": "p10"
        })
        rand["Model"] = "Random Recommender"

        df = pd.concat([
            salsa[["dataset", "seed", "Model", "fairness", "p10"]],
            ss[["dataset", "seed", "Model", "fairness", "p10"]],
            tail[["dataset", "seed", "Model", "fairness", "p10"]],
            cn[["dataset", "seed", "Model", "fairness", "p10"]],
            rand[["dataset", "seed", "Model", "fairness", "p10"]]
        ], ignore_index=True)
        
        print("Data processing complete.")
        return df

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Please check your input paths.")
        return pd.DataFrame() # Return an empty DataFrame on error

def create_plot(df, output_dir, output_filename):
    """Generates and saves the plot from the processed DataFrame."""
    print("Generating plot...")
    fig, axes = plt.subplots(1, 4, figsize=(34, 7), sharex=True, sharey=True)
    sns.set(style="whitegrid", font_scale=2)

    marker_list = ["o", "s", "D", "^", "v", "P", "X", "*"]
    color_palette = sns.color_palette("tab10", len(marker_list))

    for idx, (ax, ds) in enumerate(zip(axes, ["twitter", "pokec", "wiki", "reed98"])):
        sub = df[df["dataset"] == ds]
        models = sub["Model"].unique()

        for i, (model, marker) in enumerate(zip(models, marker_list)):
            model_data = sub[sub["Model"] == model]
            fair_mean, p10_mean = model_data["fairness"].mean(), model_data["p10"].mean()
            fair_std, p10_std = model_data["fairness"].std(), model_data["p10"].std()

            ax.errorbar(fair_mean, p10_mean, xerr=fair_std, yerr=p10_std,
                        fmt=marker, markersize=14, capsize=8, capthick=2,
                        label=model, color=color_palette[i], mec='black', mew=0.5)

        ax.set_title(ds.capitalize(), fontsize=30)
        ax.set_xlabel("Fairness (1 – Gini)", fontsize=30)
        if idx == 0:
            ax.set_ylabel("Precision @ 10", fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.legend(loc="best", fontsize=19.5)

    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to: {output_path}")

def main():
    """The main entry point of the script."""
    # Step 1: Get user inputs
    args = parse_arguments()
    
    # Step 2: Load and wrangle the data
    processed_df = load_and_process_data(args)
    
    # Step 3: Create the plot if data was loaded successfully
    if not processed_df.empty:
        create_plot(processed_df, args.output_dir, args.output_filename)

if __name__ == '__main__':
    main()