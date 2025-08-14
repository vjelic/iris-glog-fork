# utils/reporting.py

import json
import os
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
import torch

def print_benchmark_header(config: Any, num_ranks: int):
    """Prints a formatted header for a new benchmark suite."""
    console = Console()
    header_table = Table.grid(expand=True)
    header_table.add_column(style="bold white on blue")
    header_table.add_row(f"### New Benchmark Suite: Fused Attention ###")
    header_table.add_row(f"# Config: Batch={config.batch_size}, Q-Heads={config.num_q_heads}, Head-Dim={config.head_size}")
    header_table.add_row(f"# Num GPUs: {num_ranks}")
    console.print(header_table)

def print_correctness_comparison(computed: torch.Tensor, reference: torch.Tensor, rank: int):
    """Prints a detailed side-by-side comparison for correctness checking."""
    if rank != 0:
        return

    console = Console()
    console.print("\n--- Detailed Validation on Rank 0 ---", style="bold yellow")

    table = Table(title="Comparison of First 16 Values (Head 0, Sequence 0)")
    table.add_column("Index", style="cyan")
    table.add_column("Computed", style="magenta")
    table.add_column("Reference", style="green")
    table.add_column("Abs. Difference", style="red")

    comp_slice = computed[0, 0, :16].cpu().float()
    ref_slice = reference[0, 0, :16].cpu().float()
    diff_slice = torch.abs(comp_slice - ref_slice)

    for i in range(len(comp_slice)):
        table.add_row(
            f"{i}",
            f"{comp_slice[i]:.6f}",
            f"{ref_slice[i]:.6f}",
            f"{diff_slice[i]:.6f}",
        )
    
    console.print(table)
    max_diff = torch.max(torch.abs(computed - reference))
    console.print(f"\n[bold]Max absolute difference across all elements:[/bold] [red]{max_diff:.6f}[/red]")


def print_performance_summary(results_summary: List[Dict[str, Any]], config: Any, num_ranks: int):
    """Prints a final summary table of performance benchmarks."""
    console = Console()
    
    title = (
        f"Performance Summary: "
        f"B={config.batch_size}, Q_H={config.num_q_heads}, H_DIM={config.head_size}, GPUs={num_ranks}"
    )
    
    table = Table(title=title, style="bold", title_style="white")
    table.add_column("KV Len/GPU", justify="right", style="cyan")
    table.add_column("Cache (GB)", justify="right", style="magenta")
    table.add_column("Time (ms)", justify="right", style="green")

    for result in results_summary:
        table.add_row(
            f"{result['kv_len']}",
            f"{result['cache_gb']:.2f}",
            f"{result['fused_full_ms']:.3f}",
        )
    
    console.print(table)


def save_results_to_json(all_benchmark_data: List[Dict[str, Any]], filename: str):
    """Appends benchmark data to a JSON file, creating it if it doesn't exist."""
    console = Console()
    
    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                console.print(f"[yellow]Warning: Existing {filename} is not a list. Overwriting.[/yellow]")
                existing_data = []
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[red]Warning: Could not read or parse {filename}. Overwriting. Error: {e}[/red]")
            existing_data = []

    existing_data.extend(all_benchmark_data)

    console.print(f"\n[bold green]Appending structured results to {filename}...[/bold green]")
    try:
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
        console.print(f"[green]Successfully appended to {filename}[/green]")
    except IOError as e:
        console.print(f"[bold red]Error writing to {filename}: {e}[/bold red]")

