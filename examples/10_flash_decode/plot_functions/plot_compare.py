import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_melt_data(filename="final_report.json"):
    """Loads and prepares the data in a 'tidy' format suitable for Seaborn."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please run the report script first.")
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
                'kv_len': result.get('kv_len'),
                **result
            }
            all_records.append(record)
    
    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    
    id_vars = ['kv_len', 'batch_size', 'num_q_heads', 'head_dim', 'num_gpus']
    value_vars = ['fused_ms', 'fused_full_ms', 'no_wait_ms', 'iris_ag_ms', 'rccl_ms']
    value_vars = [var for var in value_vars if var in df.columns]
    
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Implementation',
        value_name='Time (ms)'
    ).dropna()

    df_melted['Implementation'] = df_melted['Implementation'].str.replace('_ms', '').str.replace('_', ' ').str.title()
    df_melted['KV Label'] = df_melted['kv_len'].apply(
        lambda l: f"{int(l/1000)}K" if l < 1000000 else f"{int(l/1000000)}M"
    )
    
    return df_melted


def create_bar_plots(df):
    """Creates a bar plot for each configuration, comparing implementations."""
    if df.empty:
        print("No data available to plot.")
        return

    output_dir = "implementation_comparison"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group data to generate a separate plot file for each distinct scenario
    config_groups = df.groupby(['num_q_heads', 'head_dim', 'batch_size', 'num_gpus'])

    for config, group in config_groups:
        qh, hd, bs, gpus = config
        
        print(f"Generating plot for config: Q-Heads={qh}, Head-Dim={hd}, Batch-Size={bs}, GPUs={gpus}...")

        plt.figure(figsize=(12, 7))
        
        # Sort by kv_len to ensure x-axis is ordered
        group = group.sort_values('kv_len')

        plot = sns.barplot(
            data=group,
            x='KV Label',
            y='Time (ms)',
            hue='Implementation',
            palette='deep'
        )

        plot.set_title(
            f'Implementation Performance\n(Batch: {bs}, Q-Heads: {qh}, Head-Dim: {hd}, GPUs: {gpus})',
            fontsize=16,
            fontweight='bold'
        )
        plot.set_xlabel('Local KV Length', fontsize=12)
        plot.set_ylabel('Latency (ms)', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Implementation')
        plt.tight_layout()

        filename = os.path.join(output_dir, f"impl_compare_qh{qh}_hd{hd}_bs{bs}_gpus{gpus}.png")
        g = plot.get_figure()
        g.savefig(filename, dpi=150)
        print(f"-> Saved plot to '{filename}'")
        plt.close()


if __name__ == "__main__":
    dataframe = load_and_melt_data()
    create_bar_plots(dataframe)