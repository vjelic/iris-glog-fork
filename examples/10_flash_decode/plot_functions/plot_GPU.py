import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_and_prepare_data(filename="compare_results.json"):
    """Loads and flattens the JSON data into a pandas DataFrame."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please make sure it's in the same directory.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{filename}'. The file may be malformed.")
        return pd.DataFrame()

    all_records = []
    for suite in data:
        config = suite.get('config', {})
        for result in suite.get('results', []):
            record = {
                'batch_size': config.get('batch_size'),
                'num_q_heads': config.get('num_q_heads'),
                'head_dim': config.get('head_dim'),
                'num_gpus': config.get('num_gpus'),
                'kv_len_per_gpu': result.get('kv_len'),
                'latency_ms': result.get('fused_ms')
            }
            if record['latency_ms'] is not None:
                all_records.append(record)

    return pd.DataFrame(all_records)

def create_plots(df):
    """Creates and saves a plot for each unique configuration found in the DataFrame."""
    if df.empty:
        print("Dataframe is empty. No data to plot.")
        return

    # --- Create the output directory ---
    output_dir = "global_latency"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set_theme(style="whitegrid")
    
    config_groups = df.groupby(['batch_size', 'num_q_heads', 'head_dim'])
    plot_generated = False

    for config, group in config_groups:
        bs, qh, hd = config
        
        plot_df = group.copy()
        plot_df['Global KV Length'] = plot_df['kv_len_per_gpu'] * plot_df['num_gpus']
        plot_df['Latency (us)'] = plot_df['latency_ms'] * 1000
        plot_df['GPUs'] = plot_df['num_gpus'].astype(str) + " GPUs"
        plot_df['Global KV Label'] = plot_df['Global KV Length'].apply(
            lambda l: f"{int(l/1000)}K" if l < 1000000 else f"{int(l/1000000)}M"
        )
        
        counts_per_label = plot_df.groupby('Global KV Label')['num_gpus'].nunique()
        valid_labels = counts_per_label[counts_per_label == 4].index.tolist()
        filtered_df = plot_df[plot_df['Global KV Label'].isin(valid_labels)]
        
        if filtered_df.empty:
            print(f"Skipping plot for config (Batch={bs}, Q-Heads={qh}, Head-Dim={hd}) - no Global KV Lengths with complete data for 1, 2, 4, and 8 GPUs.")
            continue

        print(f"Generating plot for config: Batch={bs}, Q-Heads={qh}, Head-Dim={hd}...")
        plot_generated = True

        plt.figure(figsize=(14, 8))
        
        filtered_df = filtered_df.sort_values('Global KV Length')

        # --- NEW: Define the explicit order for the bars ---
        hue_order = ["1 GPUs", "2 GPUs", "4 GPUs", "8 GPUs"]

        plot = sns.barplot(
            data=filtered_df,
            x='Global KV Label',
            y='Latency (us)',
            hue='GPUs',
            palette='CMRmap',
            hue_order=hue_order # <-- Add this parameter
        )
        
        plot.set_title(
            f'Strong Scaling Latency\n(Batch: {bs}, Q-Heads: {qh}, Head-Dim: {hd})',
            fontsize=16,
            fontweight='bold'
        )
        plot.set_xlabel('Global KV Length', fontsize=12)
        plot.set_ylabel('Latency (us)', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Num GPUs')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"barplot_global_kv_bs{bs}_qh{qh}_hd{hd}.png")
        plt.savefig(filename, dpi=150)
        print(f"-> Saved bar plot to '{filename}'")
        plt.close()
    
    if not plot_generated:
        print("\nNo plots were generated. Please check your JSON file for valid data.")

if __name__ == "__main__":
    dataframe = load_and_prepare_data()
    create_plots(dataframe)