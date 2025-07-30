import json
from collections import defaultdict
import datetime

def load_json_data(filename):
    """Safely loads data from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Log file '{filename}' not found. Skipping.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Could not parse '{filename}'. File might be empty or malformed. Skipping.")
        return []

def merge_results(iris_data, rccl_data):
    """Merges results from both benchmark suites into a single structure."""
    
    merged = defaultdict(lambda: {'timings': defaultdict(dict), 'num_gpus': None})

    for suite in iris_data:
        config = suite.get('config', {})
        config_key = (
            config.get('batch_size'),
            config.get('num_q_heads'),
            config.get('head_dim')
        )
        if None in config_key:
            continue
        
        merged[config_key]['num_gpus'] = config.get('num_gpus')

        for result in suite.get('results', []):
            kv_len = result.get('kv_len')
            if kv_len:
                merged[config_key]['timings'][kv_len]['fused_ms'] = result.get('fused_ms')
                merged[config_key]['timings'][kv_len]['fused_full_ms'] = result.get('fused_full_ms')
                merged[config_key]['timings'][kv_len]['no_wait_ms'] = result.get('no_wait_ms')
                merged[config_key]['timings'][kv_len]['iris_ag_ms'] = result.get('iris_ag_ms')

    for suite in rccl_data:
        config = suite.get('config', {})
        config_key = (
            config.get('batch_size'),
            config.get('num_q_heads'),
            config.get('head_dim')
        )
        if None in config_key:
            continue
            
        if merged[config_key]['num_gpus'] is None:
            merged[config_key]['num_gpus'] = config.get('num_gpus')

        for result in suite.get('results', []):
            kv_len = result.get('kv_len')
            if kv_len:
                merged[config_key]['timings'][kv_len]['rccl_ms'] = result.get('time_ms')

    return merged

def print_and_save_report(results, text_filename=None):
    """Prints the final report to console and saves it to a text file."""
    if not results:
        print("No valid benchmark data found to report.")
        return

    report_lines = []
    
    for config, data in sorted(results.items()):
        batch, q_heads, h_dim = config
        num_gpus = data.get('num_gpus', 'N/A')
        kv_data = data.get('timings', {})
        
        # --- NEW: Create a temporary list to hold only rows with complete data ---
        complete_rows = []
        
        for kv_len, timings in sorted(kv_data.items()):
            fused_ms = timings.get('fused_ms')
            ff_ms = timings.get('fused_full_ms')
            nw_ms = timings.get('no_wait_ms')
            ag_ms = timings.get('iris_ag_ms')
            rccl_ms = timings.get('rccl_ms')

            # --- NEW: Check if all 5 key data points are present ---
            required_timings = [fused_ms, ff_ms, nw_ms, ag_ms, rccl_ms]
            if not all(isinstance(t, float) for t in required_timings):
                continue # Skip this row if any data is missing
            
            # If we reach here, the row is complete
            complete_rows.append((kv_len, timings))
        
        if not complete_rows:
            continue # Skip this whole table if no complete rows were found

        sep = '='*130
        report_lines.extend([f"\n\n{sep}", f"### Combined Report: Batch={batch}, Q-Heads={q_heads}, Head-Dim={h_dim}, GPUs={num_gpus} ###", f"{sep}"])
        
        table_header = (f"{'KV Length':<12} | {'Fused (ms)':<15} | {'Fused Full (ms)':<18} | {'No Wait (ms)':<15} | "
                        f"{'Iris AG (ms)':<15} | {'RCCL AG (ms)':<15} | {'Speedup FF/RCCL':<18} | {'Speedup FF/F':<15}")
        table_sep = "-" * len(table_header)
        report_lines.extend([table_header, table_sep])

        for kv_len, timings in complete_rows:
            # (Timings are already fetched, just need to re-assign for formatting)
            fused_ms, ff_ms, nw_ms, ag_ms, rccl_ms = [timings.get(k) for k in ['fused_ms', 'fused_full_ms', 'no_wait_ms', 'iris_ag_ms', 'rccl_ms']]
            
            valid_timings = [fused_ms, ff_ms, nw_ms, ag_ms, rccl_ms]
            min_time = min(valid_timings)

            def format_time(time_val):
                marker = " *" if abs(time_val - min_time) < 1e-9 else ""
                return f"{time_val:.3f}{marker}"

            fused_str, ff_str, nw_str, ag_str, rccl_str = map(format_time, valid_timings)
            
            speedup_ff_rccl_str = f"{(rccl_ms / ff_ms):.2f}x"
            speedup_ff_f_str = f"{(fused_ms / ff_ms):.2f}x"

            row = (f"{kv_len:<12} | {fused_str:>15} | {ff_str:>18} | {nw_str:>15} | {ag_str:>15} | {rccl_str:>15} | "
                   f"{speedup_ff_rccl_str:>18} | {speedup_ff_f_str:>15}")
            report_lines.append(row)
        
        report_lines.append(table_sep)
    
    for line in report_lines:
        print(line)

    if text_filename:
        try:
            with open(text_filename, 'a') as f:
                f.write(f"\n\n{'#'*40} Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'#'*40}\n")
                f.write('\n'.join(report_lines))
            print(f"\nReport successfully appended to {text_filename}")
        except IOError as e:
            print(f"\nError writing text report to {text_filename}: {e}")

def save_merged_json(results, json_filename=None):
    """Saves the merged data structure to a new JSON file, filtering for complete data."""
    if not results or not json_filename:
        return

    output_list = []
    for config_key, data in sorted(results.items()):
        batch, q_heads, h_dim = config_key
        
        # --- NEW: Filter results to only include rows with complete data ---
        complete_results = []
        for kv, timings in sorted(data.get('timings', {}).items()):
            required_keys = ['fused_ms', 'fused_full_ms', 'no_wait_ms', 'iris_ag_ms', 'rccl_ms']
            if all(isinstance(timings.get(key), float) for key in required_keys):
                complete_results.append({'kv_len': kv, **timings})

        if not complete_results:
            continue

        suite = {
            "config": {
                "batch_size": batch, "num_q_heads": q_heads, "head_dim": h_dim, "num_gpus": data.get('num_gpus')
            },
            "results": complete_results
        }
        output_list.append(suite)
        
    try:
        with open(json_filename, 'w') as f:
            json.dump(output_list, f, indent=4)
        print(f"Merged JSON data (complete results only) saved to {json_filename}")
    except IOError as e:
        print(f"Error writing JSON report to {json_filename}: {e}")


if __name__ == "__main__":
    iris_results = load_json_data("compare_results.json")
    rccl_results = load_json_data("rccl_perf_results.json")
    
    combined_data = merge_results(iris_results, rccl_results)
    
    print_and_save_report(combined_data, text_filename="final_report.txt")
    
    save_merged_json(combined_data, json_filename="final_report.json")