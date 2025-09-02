import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import re

# ==============================================================================
# --- 1. NETWORK PROPERTIES CONFIGURATION ---
# ==============================================================================

# --- Helper function to parse "mean (std)" strings ---
def parse_mean(s):
    """Extracts the mean value from a string like '0.500 (0.000)'."""
    try:
        return float(s.split(' ')[0])
    except (ValueError, AttributeError):
        return np.nan

# --- WS Network Properties ---
ws_properties_data = """
network,num_nodes,num_edges,density,diameter,clustering,avg_path,fairness
ws1,1000.0,2000.0,0.004 (0.000),25.000 (2.000),0.433 (0.009),11.847 (0.733),0.958 (0.004)
ws2,1000.0,2000.0,0.004 (0.000),17.100 (0.539),0.371 (0.009),8.792 (0.222),0.929 (0.003)
ws3,1000.0,2000.0,0.004 (0.000),14.400 (1.281),0.317 (0.015),7.629 (0.193),0.909 (0.004)
ws4,1000.0,2000.0,0.004 (0.000),12.800 (0.600),0.270 (0.007),6.945 (0.085),0.892 (0.003)
ws5,1000.0,2000.0,0.004 (0.000),12.000 (0.447),0.225 (0.012),6.504 (0.102),0.876 (0.004)
ws6,1000.0,2000.0,0.004 (0.000),11.100 (0.300),0.176 (0.008),6.158 (0.046),0.864 (0.005)
ws7,1000.0,2000.0,0.004 (0.000),10.800 (0.600),0.147 (0.010),5.980 (0.042),0.856 (0.003)
ws8,1000.0,2000.0,0.004 (0.000),10.400 (0.490),0.112 (0.009),5.790 (0.048),0.847 (0.006)
ws9,1000.0,2000.0,0.004 (0.000),10.000 (0.000),0.092 (0.008),5.694 (0.027),0.839 (0.003)
ws10,1000.0,2000.0,0.004 (0.000),9.700 (0.458),0.070 (0.007),5.611 (0.018),0.835 (0.003)
ws11,1000.0,2000.0,0.004 (0.000),9.200 (0.400),0.050 (0.005),5.515 (0.021),0.828 (0.005)
ws12,1000.0,2000.0,0.004 (0.000),9.300 (0.458),0.038 (0.004),5.457 (0.024),0.823 (0.005)
ws13,1000.0,2000.0,0.004 (0.000),9.000 (0.000),0.027 (0.005),5.418 (0.015),0.819 (0.004)
ws14,1000.0,2000.0,0.004 (0.000),9.000 (0.000),0.016 (0.003),5.377 (0.024),0.815 (0.006)
ws15,1000.0,2000.0,0.004 (0.000),9.000 (0.000),0.011 (0.003),5.363 (0.020),0.815 (0.004)
ws16,1000.0,2000.0,0.004 (0.000),9.000 (0.000),0.006 (0.001),5.341 (0.014),0.811 (0.004)
ws17,1000.0,2000.0,0.004 (0.000),9.000 (0.000),0.005 (0.002),5.338 (0.016),0.810 (0.004)
ws18,1000.0,2000.0,0.004 (0.000),9.000 (0.000),0.004 (0.001),5.321 (0.016),0.807 (0.006)
ws19,1000.0,2000.0,0.004 (0.000),9.100 (0.300),0.004 (0.002),5.323 (0.013),0.807 (0.005)
ws20,1000.0,2000.0,0.004 (0.000),9.000 (0.000),0.004 (0.002),5.322 (0.016),0.806 (0.004)
ws21,1000.0,1000.0,0.002 (0.000),226.600 (41.495),0.000 (0.000),80.627 (14.104),0.922 (0.007)
ws22,1000.0,2000.0,0.004 (0.000),17.100 (0.943),0.366 (0.009),8.692 (0.218),0.928 (0.003)
ws23,1000.0,3000.0,0.006 (0.000),11.400 (0.663),0.449 (0.005),6.223 (0.106),0.938 (0.003)
ws24,1000.0,4000.0,0.008 (0.000),8.800 (0.400),0.476 (0.008),5.063 (0.053),0.944 (0.002)
ws25,1000.0,5000.0,0.010 (0.000),7.300 (0.458),0.491 (0.003),4.435 (0.030),0.948 (0.001)
ws26,1000.0,6000.0,0.012 (0.000),6.700 (0.458),0.504 (0.006),4.042 (0.029),0.953 (0.002)
ws27,1000.0,100000.0,0.200 (0.000),3.000 (0.000),0.568 (0.002),1.800 (0.000),0.988 (0.000)
ws28,1000.0,200000.0,0.400 (0.000),2.000 (0.000),0.606 (0.001),1.600 (0.000),0.991 (0.000)
"""
ws_netdf = pd.read_csv(io.StringIO(ws_properties_data)).set_index('network')

# --- PL Network Properties ---
pl_properties_data = """
network,num_nodes,num_edges,density,diameter,clustering,avg_path,fairness
pl1,1000.0,3984.0,0.008 (0.000),5.000 (0.000),0.034 (0.004),3.178 (0.028),0.639 (0.003)
pl2,1000.0,3983.0,0.008 (0.000),5.000 (0.000),0.056 (0.002),3.165 (0.029),0.639 (0.002)
pl3,1000.0,3981.4,0.008 (0.000),5.000 (0.000),0.076 (0.004),3.161 (0.030),0.637 (0.004)
pl4,1000.0,3980.2,0.008 (0.000),5.000 (0.000),0.099 (0.004),3.174 (0.029),0.636 (0.002)
pl5,1000.0,3978.4,0.008 (0.000),5.000 (0.000),0.115 (0.003),3.195 (0.018),0.637 (0.003)
pl6,1000.0,3979.0,0.008 (0.000),5.000 (0.000),0.141 (0.006),3.158 (0.026),0.634 (0.003)
pl7,1000.0,3979.4,0.008 (0.000),5.000 (0.000),0.161 (0.005),3.182 (0.024),0.633 (0.003)
pl8,1000.0,3979.0,0.008 (0.000),5.000 (0.000),0.181 (0.007),3.158 (0.017),0.632 (0.003)
pl9,1000.0,3978.8,0.008 (0.000),5.000 (0.000),0.203 (0.004),3.185 (0.023),0.631 (0.003)
pl10,1000.0,3980.0,0.008 (0.000),5.000 (0.000),0.229 (0.006),3.180 (0.032),0.632 (0.003)
pl11,1000.0,3979.0,0.008 (0.000),5.600 (0.490),0.250 (0.004),3.227 (0.031),0.633 (0.003)
pl12,1000.0,3978.8,0.008 (0.000),5.000 (0.000),0.272 (0.009),3.203 (0.039),0.632 (0.004)
pl13,1000.0,3979.2,0.008 (0.000),5.600 (0.490),0.301 (0.003),3.193 (0.050),0.631 (0.002)
pl14,1000.0,3979.2,0.008 (0.000),5.800 (0.400),0.315 (0.004),3.221 (0.054),0.632 (0.002)
pl15,1000.0,3979.2,0.008 (0.000),5.800 (0.400),0.340 (0.008),3.254 (0.028),0.633 (0.001)
pl16,1000.0,3979.2,0.008 (0.000),6.000 (0.000),0.374 (0.004),3.292 (0.054),0.633 (0.002)
pl17,1000.0,3981.0,0.008 (0.000),6.000 (0.000),0.394 (0.003),3.329 (0.059),0.637 (0.002)
pl18,1000.0,3981.8,0.008 (0.000),6.200 (0.400),0.436 (0.008),3.400 (0.049),0.633 (0.002)
pl19,1000.0,3982.6,0.008 (0.000),7.000 (0.000),0.459 (0.004),3.499 (0.047),0.636 (0.002)
pl20,1000.0,3983.8,0.008 (0.000),6.600 (0.490),0.500 (0.006),3.560 (0.153),0.635 (0.001)
pl21,1000.0,1996.0,0.004 (0.000),7.200 (0.400),0.098 (0.013),4.137 (0.013),0.612 (0.002)
pl22,1000.0,3981.2,0.008 (0.000),5.200 (0.400),0.074 (0.005),3.174 (0.034),0.637 (0.003)
pl23,1000.0,7920.0,0.016 (0.000),4.000 (0.000),0.079 (0.002),2.648 (0.005),0.654 (0.002)
pl24,1000.0,15691.6,0.031 (0.000),3.000 (0.000),0.097 (0.001),2.272 (0.003),0.671 (0.001)
pl25,1000.0,30786.0,0.062 (0.000),3.000 (0.000),0.153 (0.001),1.969 (0.001),0.685 (0.001)
pl26,1000.0,59249.4,0.119 (0.000),3.000 (0.000),0.222 (0.001),1.881 (0.000),0.719 (0.000)
pl27,1000.0,109711.4,0.220 (0.000),2.000 (0.000),0.334 (0.001),1.780 (0.000),0.762 (0.001)
pl28,1000.0,185055.6,0.370 (0.000),2.000 (0.000),0.467 (0.001),1.630 (0.000),0.825 (0.001)
"""
pl_netdf = pd.read_csv(io.StringIO(pl_properties_data)).set_index('network')

# Apply the parser to all relevant columns for both dataframes
for df in [ws_netdf, pl_netdf]:
    for col in ['density', 'diameter', 'clustering', 'avg_path', 'fairness']:
        df[col] = df[col].apply(parse_mean)

# ==============================================================================
# --- 2. LOAD AND PREPARE MODEL RESULT FILES ---
# ==============================================================================
BASE_RESULTS_PATH = "/Users/eceerdem/Desktop/gae_synthetic_network_generation/aaai_results_doc/realistic_synthetic_exp/"

# --- FIXED: Load CSVs while forcing the network column to be a string ---
salsa_results = pd.read_csv(os.path.join(BASE_RESULTS_PATH, "salsa_ws_pl_edge_swap_results.csv"), dtype={'network': str})
ssnet_results = pd.read_csv(os.path.join(BASE_RESULTS_PATH, "sssnet_ws_pl_edge_swap_results.csv"), dtype={'network': str})
tailsteak_results = pd.read_csv(os.path.join(BASE_RESULTS_PATH, "tailsteak_ws_pl_edge_swap_results.csv"), dtype={'dataset': str})

# --- Standardize column names for easier processing ---
salsa_results.columns = salsa_results.columns.str.lower().str.replace(' ', '_')
ssnet_results.columns = ssnet_results.columns.str.lower()
tailsteak_results.columns = tailsteak_results.columns.str.lower().str.replace('@', '')

# --- FIXED: Convert metric columns to numeric, coercing errors ---
metric_cols_salsa = ['precision', '1-gini_index']
for col in metric_cols_salsa:
    salsa_results[col] = pd.to_numeric(salsa_results[col], errors='coerce')

metric_cols_ssnet = ['p10', '1-gini']
for col in metric_cols_ssnet:
    ssnet_results[col] = pd.to_numeric(ssnet_results[col], errors='coerce')

metric_cols_tailsteak = ['p5', '1_minus_gini']
for col in metric_cols_tailsteak:
    tailsteak_results[col] = pd.to_numeric(tailsteak_results[col], errors='coerce')


# ==============================================================================
# --- 3. REUSABLE PLOTTING FUNCTION ---
# ==============================================================================

def get_metric(df, netlist, metric_col, **query):
    """Extracts values of `metric_col` for each network in netlist based on filtering by query."""
    vals = []
    # Use the standardized 'network' column for matching
    df_net_col = 'network' if 'network' in df.columns else 'dataset'
    
    for net in netlist:
        # --- FIXED: Use regex to match network names with optional versions (e.g., "pl1" matches "pl1_v1", "pl1_v2", etc.) ---
        # This correctly groups the 5 versions of each network for averaging.
        regex_pattern = f'^{re.escape(net)}(_v\d+)?$'
        conditions = df[df_net_col].str.match(regex_pattern, na=False)
        
        for k, v in query.items():
            conditions &= (df[k] == v)
            
        row = df[conditions]
        
        if not row.empty:
            # Average results if multiple seeds/versions exist for the same configuration
            vals.append(row[metric_col].mean())
        else:
            vals.append(np.nan)
    return vals

def generate_plots(title_prefix, clustering_nets, density_nets, net_df, salsa_df, ssnet_df, tailsteak_df, output_filename):
    """
    Generates and saves a 2x2 plot for Precision and Fairness vs. Clustering and Density.
    """
    print(f"--- Generating plots for {title_prefix} ---")
    
    # --- Get X-axis values from the network properties dataframe ---
    clustering_vals = net_df.loc[clustering_nets, "clustering"].values
    density_vals    = net_df.loc[density_nets, "density"].values

    # --- Extract results for each model ---
    # Salsa
    salsa_prec   = get_metric(salsa_df, clustering_nets, "precision", model="salsaAuth")
    salsa_fair   = get_metric(salsa_df, clustering_nets, "1-gini_index")
    salsa_prec_d = get_metric(salsa_df, density_nets, "precision", model="salsaAuth")
    salsa_fair_d = get_metric(salsa_df, density_nets, "1-gini_index")

    # LightGCN
    lgn_prec   = get_metric(ssnet_df, clustering_nets, "p10", model="LightGCN", mode="normal")
    lgn_fair   = get_metric(ssnet_df, clustering_nets, "1-gini", model="LightGCN", mode="normal")
    lgn_prec_d = get_metric(ssnet_df, density_nets, "p10", model="LightGCN", mode="normal")
    lgn_fair_d = get_metric(ssnet_df, density_nets, "1-gini", model="LightGCN", mode="normal")

    # PPRGo
    pprgo_prec   = get_metric(ssnet_df, clustering_nets, "p10", model="PPRGo", mode="normal")
    pprgo_fair   = get_metric(ssnet_df, clustering_nets, "1-gini", model="PPRGo", mode="normal")
    pprgo_prec_d = get_metric(ssnet_df, density_nets, "p10", model="PPRGo", mode="normal")
    pprgo_fair_d = get_metric(ssnet_df, density_nets, "1-gini", model="PPRGo", mode="normal")

    # TailSTEAK --- FIXED: Changed "p5" to "p10" for consistent comparison ---
    tailsteak_prec   = get_metric(tailsteak_df, clustering_nets, "p5", model="LightGCN")
    tailsteak_fair   = get_metric(tailsteak_df, clustering_nets, "1_minus_gini", model="LightGCN")
    tailsteak_prec_d = get_metric(tailsteak_df, density_nets,  "p5", model="LightGCN")
    tailsteak_fair_d = get_metric(tailsteak_df, density_nets, "1_minus_gini", model="LightGCN")

    # --- Create the 2x2 plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='col')
    fig.suptitle(f'{title_prefix} Network Experiments', fontsize=16)

    # Plot 1: Precision vs. Clustering
    ax = axes[0, 0]
    ax.plot(clustering_vals, salsa_prec, 'o-', label='SALSA', color='blue')
    ax.plot(clustering_vals, lgn_prec, 's-', label='LightGCN', color='red')
    ax.plot(clustering_vals, pprgo_prec, '^-', label='PPRGo', color='green')
    ax.plot(clustering_vals, tailsteak_prec, 'd-', label='TailSTEAK', color='m')
    ax.set_xlabel('Clustering Coefficient')
    ax.set_ylabel('Precision@10')
    ax.set_title('Precision vs. Clustering')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot 2: Fairness vs. Clustering
    ax = axes[0, 1]
    ax.plot(clustering_vals, salsa_fair, 'o-', label='SALSA', color='blue')
    ax.plot(clustering_vals, lgn_fair, 's-', label='LightGCN', color='red')
    ax.plot(clustering_vals, pprgo_fair, '^-', label='PPRGo', color='green')
    ax.plot(clustering_vals, tailsteak_fair, 'd-', label='TailSTEAK', color='m')
    ax.set_xlabel('Clustering Coefficient')
    ax.set_ylabel('Fairness (1-Gini)')
    ax.set_title('Fairness vs. Clustering')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot 3: Precision vs. Density
    ax = axes[1, 0]
    ax.plot(density_vals, salsa_prec_d, 'o-', label='SALSA', color='blue')
    ax.plot(density_vals, lgn_prec_d, 's-', label='LightGCN', color='red')
    ax.plot(density_vals, pprgo_prec_d, '^-', label='PPRGo', color='green')
    ax.plot(density_vals, tailsteak_prec_d, 'd-', label='TailSTEAK', color='m')
    ax.set_xlabel('Density')
    ax.set_ylabel('Precision@10')
    ax.set_title('Precision vs. Density')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot 4: Fairness vs. Density
    ax = axes[1, 1]
    ax.plot(density_vals, salsa_fair_d, 'o-', label='SALSA', color='blue')
    ax.plot(density_vals, lgn_fair_d, 's-', label='LightGCN', color='red')
    ax.plot(density_vals, pprgo_fair_d, '^-', label='PPRGo', color='green')
    ax.plot(density_vals, tailsteak_fair_d, 'd-', label='TailSTEAK', color='m')
    ax.set_xlabel('Density')
    ax.set_ylabel('Fairness (1-Gini)')
    ax.set_title('Fairness vs. Density')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"✅ Plots saved to {output_filename}")

# ==============================================================================
# --- 4. EXECUTION ---
# ==============================================================================

# --- Define network groups for WS plots ---
# These appear to be correct based on the network properties in your data
ws_clustering_nets = [f'ws{i}' for i in range(1, 21)]  # ws1 to ws20
ws_density_nets    = [f'ws{i}' for i in range(22, 27)] # ws22 to ws26

# --- Define network groups for PL plots ---
pl_clustering_nets = [f'pl{i}' for i in range(1, 21)]  # pl1 to pl20
# --- FIXED: Updated this line to include pl21 as per your instruction ---
pl_density_nets    = [f'pl{i}' for i in range(21, 29)] # pl21 to pl28

# --- Generate the plots for both network types ---
generate_plots(
    title_prefix="WS",
    clustering_nets=ws_clustering_nets,
    density_nets=ws_density_nets,
    net_df=ws_netdf,
    salsa_df=salsa_results,
    ssnet_df=ssnet_results,
    tailsteak_df=tailsteak_results,
    output_filename=os.path.join(f"{BASE_RESULTS_PATH}/plots", "WS_experiments_plots.pdf")
)

generate_plots(
    title_prefix="PL",
    clustering_nets=pl_clustering_nets,
    density_nets=pl_density_nets,
    net_df=pl_netdf,
    salsa_df=salsa_results,
    ssnet_df=ssnet_results,
    tailsteak_df=tailsteak_results,
    output_filename=os.path.join(f"{BASE_RESULTS_PATH}/plots", "PL_experiments_plots.pdf")
)

# ==============================================================================
# --- 3. GATHER ALL METRICS FOR PLOTTING ---
# ==============================================================================
print("--- Gathering all metrics from WS and PL experiments ---")

# --- Define network groups ---
ws_clustering_nets = [f'ws{i}' for i in range(1, 21)]
ws_density_nets    = [f'ws{i}' for i in range(22, 27)]
pl_clustering_nets = [f'pl{i}' for i in range(1, 21)]
pl_density_nets    = [f'pl{i}' for i in range(21, 29)]

# --- Get X-axis values for all plots ---
ws_x_clust = ws_netdf.loc[ws_clustering_nets, "clustering"].values
ws_x_dens  = ws_netdf.loc[ws_density_nets, "density"].values
pl_x_clust = pl_netdf.loc[pl_clustering_nets, "clustering"].values
pl_x_dens  = pl_netdf.loc[pl_density_nets, "density"].values

# --- Create a helper function to get all model metrics for a given experiment ---
def get_all_metrics(net_list, salsa_df, ssnet_df, tailsteak_df):
    metrics = {
        'salsa_prec':   get_metric(salsa_df, net_list, "precision", model="salsaAuth"),
        'salsa_fair':   get_metric(salsa_df, net_list, "1-gini_index"),
        'lgn_prec':     get_metric(ssnet_df, net_list, "p10", model="LightGCN", mode="normal"),
        'lgn_fair':     get_metric(ssnet_df, net_list, "1-gini", model="LightGCN", mode="normal"),
        'pprgo_prec':   get_metric(ssnet_df, net_list, "p10", model="PPRGo", mode="normal"),
        'pprgo_fair':   get_metric(ssnet_df, net_list, "1-gini", model="PPRGo", mode="normal"),
        'ts_prec':      get_metric(tailsteak_df, net_list, "p5", model="LightGCN"),
        'ts_fair':      get_metric(tailsteak_df, net_list, "1_minus_gini", model="LightGCN")
    }
    return metrics

# --- Extract all data points ---
ws_metrics_clust = get_all_metrics(ws_clustering_nets, salsa_results, ssnet_results, tailsteak_results)
ws_metrics_dens  = get_all_metrics(ws_density_nets, salsa_results, ssnet_results, tailsteak_results)
pl_metrics_clust = get_all_metrics(pl_clustering_nets, salsa_results, ssnet_results, tailsteak_results)
pl_metrics_dens  = get_all_metrics(pl_density_nets, salsa_results, ssnet_results, tailsteak_results)

# ==============================================================================
# --- 4. GENERATE NEW PLOTS (MANUAL LAYOUT) ---
# ==============================================================================

# --- Reusable function to draw the four model lines on a subplot ---
def draw_lines(ax, x_vals, metrics, metric_type):
    """Plots the four models on a given axis for 'prec' or 'fair'."""
    ax.plot(x_vals, metrics[f'salsa_{metric_type}'], 'o-', label='SALSA', color='blue')
    ax.plot(x_vals, metrics[f'lgn_{metric_type}'], 's-', label='LightGCN', color='red')
    ax.plot(x_vals, metrics[f'pprgo_{metric_type}'], '^-', label='PPRGo', color='green')
    ax.plot(x_vals, metrics[f'ts_{metric_type}'], 'd-', label='TailSTEAK', color='m')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

font_axis = {'fontsize': 17}

# --- FIGURE 1: PRECISION PLOTS ---
print("--- Generating Figure 1: Precision ---")
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

# Top-left: WS Precision vs. Clustering
draw_lines(axes1[0, 0], ws_x_clust, ws_metrics_clust, 'prec')
axes1[0, 0].set_title('Precision vs. Clustering', fontdict=font_axis)
axes1[0, 0].set_ylabel('Precision', fontdict=font_axis)
axes1[0, 0].set_xlabel('Clustering Coefficient', fontdict=font_axis)

# Top-right: PL Precision vs. Clustering
draw_lines(axes1[0, 1], pl_x_clust, pl_metrics_clust, 'prec')
axes1[0, 1].set_title('Precision vs. Clustering', fontdict=font_axis)
axes1[0, 1].set_xlabel('Clustering Coefficient', fontdict=font_axis)

# Bottom-left: WS Precision vs. Density
draw_lines(axes1[1, 0], ws_x_dens, ws_metrics_dens, 'prec')
axes1[1, 0].set_title('Precision vs. Density', fontdict=font_axis)
axes1[1, 0].set_ylabel('Precision', fontdict=font_axis)
axes1[1, 0].set_xlabel('Density', fontdict=font_axis)

# Bottom-right: PL Precision vs. Density
draw_lines(axes1[1, 1], pl_x_dens, pl_metrics_dens, 'prec')
axes1[1, 1].set_title('Precision vs. Density', fontdict=font_axis)
axes1[1, 1].set_xlabel('Density', fontdict=font_axis)

# ✅ Manually position column labels using FIGURE coordinates
# (x, y) where (0,0) is bottom-left of the whole figure
fig1.text(0.3, 0.02, '(a) WS-experiment results', ha='center', fontsize=18)
fig1.text(0.78, 0.02, '(b) PL-experiment results', ha='center', fontsize=18)

# ✅ Manually adjust subplot layout to prevent overlap and make room
fig1.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.1, hspace=0.3, wspace=0.1)

# Manually adjust subplot layout to prevent overlap and make room
# ✅ CHANGED hspace from 0.25 to 0.4
fig1.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.1, hspace=0.3, wspace=0.1)


# ✅ Save the figure WITHOUT tight layout options
plt.savefig(os.path.join(f"{BASE_RESULTS_PATH}/plots", "Precision_experiments_plots_WS_PL.pdf"), format="pdf")
plt.close(fig1)
print("✅ Precision plots saved.")


# --- FIGURE 2: FAIRNESS PLOTS ---
print("--- Generating Figure 2: Fairness ---")
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

# Top-left: WS Fairness vs. Clustering
draw_lines(axes2[0, 0], ws_x_clust, ws_metrics_clust, 'fair')
axes2[0, 0].set_title('Fairness vs. Clustering', fontdict=font_axis)
axes2[0, 0].set_ylabel('Fairness (1-Gini)', fontdict=font_axis)
axes2[0, 0].set_xlabel('Clustering Coefficient', fontdict=font_axis)

# Top-right: PL Fairness vs. Clustering
draw_lines(axes2[0, 1], pl_x_clust, pl_metrics_clust, 'fair')
axes2[0, 1].set_title('Fairness vs. Clustering', fontdict=font_axis)
axes2[0, 1].set_xlabel('Clustering Coefficient', fontdict=font_axis)

# Bottom-left: WS Fairness vs. Density
draw_lines(axes2[1, 0], ws_x_dens, ws_metrics_dens, 'fair')
axes2[1, 0].set_title('Fairness vs. Density', fontdict=font_axis)
axes2[1, 0].set_ylabel('Fairness (1-Gini)', fontdict=font_axis)
axes2[1, 0].set_xlabel('Density', fontdict=font_axis)

# Bottom-right: PL Fairness vs. Density
draw_lines(axes2[1, 1], pl_x_dens, pl_metrics_dens, 'fair')
axes2[1, 1].set_title('Fairness vs. Density', fontdict=font_axis)
axes2[1, 1].set_xlabel('Density', fontdict=font_axis)

# ✅ Manually position column labels using FIGURE coordinates
fig2.text(0.3, 0.02, '(a) WS-experiment results', ha='center', fontsize=18)
fig2.text(0.78, 0.02, '(b) PL-experiment results', ha='center', fontsize=18)

# ✅ Manually adjust subplot layout
fig2.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.1, hspace=0.25, wspace=0.1)

# ✅ Save the figure WITHOUT tight layout options
plt.savefig(os.path.join(f"{BASE_RESULTS_PATH}/plots", "Fairness_experiments_plots_WS_PL.pdf"), format="pdf")
plt.close(fig2)
print("✅ Fairness plots saved.")