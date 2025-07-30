import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

def load_and_prepare_data(filename="compare_results.json"):
    """Loads JSON data and calculates bandwidth metrics."""
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
            latency_ms = result.get('fused_full_ms')
            if latency_ms is None or latency_ms == 0:
                continue

            record = {
                'batch_size': config.get('batch_size'),
                'num_q_heads': config.get('num_q_heads'),
                'head_dim': config.get('head_dim'),
                'num_gpus': config.get('num_gpus'),
                'kv_len_per_gpu': result.get('kv_len'),
                'latency_ms': latency_ms
            }
            all_records.append(record)

    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    # --- Calculate Metrics ---
    df['num_kv_heads'] = df['num_q_heads'] // 8
    df['total_bytes_read_local'] = df['kv_len_per_gpu'] * df['num_kv_heads'] * df['head_dim'] * 2 * 2
    df['Local Bandwidth (GB/s)'] = (df['total_bytes_read_local'] / (df['latency_ms'] / 1000)) / 1e9
    df['Global KV Length'] = df['kv_len_per_gpu'] * df['num_gpus']
    
    return df

def create_plots(df):
    """Creates and saves a strong scaling plot for each unique configuration."""
    if df.empty:
        print("Dataframe is empty. No data to plot.")
        return

    output_dir = "bandwidth_global"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sns.set_theme(style="whitegrid")
    config_groups = df.groupby(['batch_size', 'num_q_heads', 'head_dim'])
    plot_generated = False

    for config, group in config_groups:
        bs, qh, hd = config
        
        if group.empty:
            continue

        print(f"Generating plot for config: Batch={bs}, Q-Heads={qh}, Head-Dim={hd}...")
        plot_generated = True

        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        
        style_map = {
            1:  {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
            2:  {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
            4:  {'color': '#2ca02c', 'marker': '^', 'linestyle': '-'},
            8:  {'color': '#d62728', 'marker': 'D', 'linestyle': '-'},
            16: {'color': '#9467bd', 'marker': 'X', 'linestyle': '--'},
            32: {'color': '#8c564b', 'marker': 'P', 'linestyle': '--'},
        }

        unique_gpus = sorted(group['num_gpus'].unique())
        for gpu_count in unique_gpus:
            subset = group[group['num_gpus'] == gpu_count]
            style = style_map.get(gpu_count, {})
            
            sns.lineplot(
                data=subset,
                x='Global KV Length',
                y='Local Bandwidth (GB/s)',
                ax=ax,
                label=f'{gpu_count} GPUs',
                marker=style.get('marker'),
                color=style.get('color'),
                linestyle=style.get('linestyle'),
                linewidth=2.5
            )
        
        ax.set_title(
            f'Strong Scaling Bandwidth\n(Batch: {bs}, Q-Heads: {qh}, Head-Dim: {hd})',
            fontsize=16,
            fontweight='bold'
        )
        ax.set_xlabel('Global KV Length', fontsize=12)
        ax.set_ylabel('Local Bandwidth (GB/s)', fontsize=12)
        ax.set_ylim(bottom=0)
        
        # --- NEW: Set the maximum limit for the x-axis ---
        ax.set_xlim(right=2000000)
        
        def custom_formatter(x, pos):
            if x >= 1e6:
                return f'{x/1e6:.1f}M'
            if x >= 1e3:
                return f'{int(x/1e3)}K'
            return str(int(x))
        
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))
        plt.xticks(rotation=30, ha='right')
        
        ax.legend(title='Num GPUs')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"strong_scaling_bs{bs}_qh{qh}_hd{hd}.png")
        plt.savefig(filename, dpi=150)
        print(f"-> Saved plot to '{filename}'")
        plt.close()
    
    if not plot_generated:
        print("\nNo plots were generated. Please check your JSON file for valid data.")

if __name__ == "__main__":
    dataframe = load_and_prepare_data()
    create_plots(dataframe)