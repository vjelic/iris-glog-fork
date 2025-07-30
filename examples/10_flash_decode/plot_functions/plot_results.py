import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(iris_file="compare_results.json", rccl_file="rccl_perf_results.json"):
    """Loads and merges data from JSON files into a single pandas DataFrame."""
    
    def load_json(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Log file '{filename}' not found.")
            return []
        except json.JSONDecodeError:
            print(f"Warning: Could not parse '{filename}'.")
            return []

    iris_data = load_json(iris_file)
    rccl_data = load_json(rccl_file)
    
    all_records = []

    # Process Iris data
    for suite in iris_data:
        config = suite.get('config', {})
        for result in suite.get('results', []):
            record = {
                'batch_size': config.get('batch_size'),
                'num_q_heads': config.get('num_q_heads'),
                'head_dim': config.get('head_dim'),
                'num_gpus': config.get('num_gpus'),
                'kv_len': result.get('kv_len'),
                **result  # Add all timings like fused_ms, iris_ag_ms, etc.
            }
            all_records.append(record)
            
    # Process and merge RCCL data
    for suite in rccl_data:
        config = suite.get('config', {})
        for result in suite.get('results', []):
            # Find the corresponding record from the Iris data to update
            found = False
            for record in all_records:
                if (record['batch_size'] == config.get('batch_size') and
                    record['num_q_heads'] == config.get('num_q_heads') and
                    record['head_dim'] == config.get('head_dim') and
                    record['kv_len'] == result.get('kv_len')):
                    record['rccl_ms'] = result.get('time_ms')
                    found = True
                    break
            # If no matching Iris record, create a new one (less common)
            if not found:
                record = {
                    'batch_size': config.get('batch_size'),
                    'num_q_heads': config.get('num_q_heads'),
                    'head_dim': config.get('head_dim'),
                    'num_gpus': config.get('num_gpus'),
                    'kv_len': result.get('kv_len'),
                    'rccl_ms': result.get('time_ms')
                }
                all_records.append(record)

    df = pd.DataFrame(all_records)
    
    # Melt the DataFrame to make it suitable for plotting with seaborn
    return pd.melt(df, 
                   id_vars=['batch_size', 'num_q_heads', 'head_dim', 'num_gpus', 'kv_len'], 
                   value_vars=['fused_full_ms', 'iris_ag_ms', 'rccl_ms', 'fused_ms', 'no_wait_ms'],
                   var_name='Implementation', 
                   value_name='Time (ms)')


def plot_results(df):
    """Generates and saves plots for each benchmark configuration."""
    if df.empty:
        print("No data available to plot.")
        return

    # Use a nice visual style
    sns.set_theme(style="whitegrid")

    # Group data by each unique configuration
    config_groups = df.groupby(['batch_size', 'num_q_heads', 'head_dim', 'num_gpus'])

    for config, group in config_groups:
        bs, qh, hd, gpus = config
        
        plt.figure(figsize=(12, 7))
        
        plot = sns.lineplot(
            data=group,
            x='kv_len',
            y='Time (ms)',
            hue='Implementation',
            style='Implementation',
            markers=True,
            dashes=False
        )
        
        plot.set_title(
            f'Performance Comparison\n(Batch Size: {bs}, Q-Heads: {qh}, Head Dim: {hd}, GPUs: {gpus})',
            fontsize=16,
            fontweight='bold'
        )
        plot.set_xlabel('KV Cache Length per GPU', fontsize=12)
        plot.set_ylabel('Average Time (ms)', fontsize=12)
        plot.set_xscale('log', base=2) # Use a log scale for the x-axis for better visibility
        plt.legend(title='Implementation')
        plt.tight_layout()
        
        # Save the plot to a file
        filename = f"plot_bs{bs}_qh{qh}_hd{hd}.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot to '{filename}'")
        plt.close()


if __name__ == "__main__":
    # 1. Load, merge, and structure the data
    dataframe = load_and_prepare_data()
    
    # 2. Generate and save the plots
    plot_results(dataframe)