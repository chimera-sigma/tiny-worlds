#!/usr/bin/env python3
"""
Aggregate results from summary.jsonl into a CSV for plotting/analysis.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --summary experiments/results/summary.jsonl --output experiments/results/aggregated_results.csv
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_summary(summary_path: Path):
    """Load all runs from summary.jsonl."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found at: {summary_path}")
    
    runs = []
    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse line: {line[:50]}... ({e})")
    
    return runs


def derive_mode(run):
    """
    Derive experiment mode from run configuration.

    Modes:
    - readout_only: only readout layer trained (readout_only=True)
    - scrambled: scramble_core=True (destroys structure)
    - full: normal full training
    """
    readout_only = run.get('readout_only', False)
    scramble_core = run.get('scramble_core', False)

    if readout_only:
        return 'readout_only'
    elif scramble_core:
        return 'scrambled'
    else:
        return 'full'


def aggregate_runs(runs):
    """
    Group runs by (world_id, model_id, hidden_dim, mode)
    and compute aggregate statistics.

    Mode is derived from readout_only, scramble_core, and max_steps.
    """
    # Group by key
    groups = defaultdict(list)
    for run in runs:
        mode = derive_mode(run)
        key = (
            run['world_id'],
            run['model_id'],
            run['hidden_dim'],
            mode
        )
        groups[key].append(run)
    
    aggregated = []
    
    for key, group in groups.items():
        world_id, model_id, hidden_dim, mode = key
        
        # Extract first run for metadata
        first = group[0]
        
        # ΔNLL stats
        delta_nll_values = [r['delta_nll_mean'] for r in group]
        delta_nll_mean = sum(delta_nll_values) / len(delta_nll_values)
        delta_nll_min = min(delta_nll_values)
        delta_nll_max = max(delta_nll_values)
        
        # p-value (mean across runs if repeated)
        p_values = [r['delta_nll_p_perm'] for r in group]
        p_mean = sum(p_values) / len(p_values)
        
        # Probe stats (may be None/NaN)
        r2_trained_values = [r['probe_r2_trained_mean'] for r in group if r['probe_r2_trained_mean'] is not None]
        r2_random_values = [r['probe_r2_random_mean'] for r in group if r['probe_r2_random_mean'] is not None]
        r2_permuted_values = [r['probe_r2_permuted_mean'] for r in group if r['probe_r2_permuted_mean'] is not None]
        
        def safe_stats(values):
            if not values:
                return None, None, None
            return (
                sum(values) / len(values),  # mean
                min(values),                # min
                max(values)                 # max
            )
        
        r2_trained_mean, r2_trained_min, r2_trained_max = safe_stats(r2_trained_values)
        r2_random_mean, r2_random_min, r2_random_max = safe_stats(r2_random_values)
        r2_permuted_mean, r2_permuted_min, r2_permuted_max = safe_stats(r2_permuted_values)
        
        aggregated.append({
            'world_id': world_id,
            'world_type': first['world_type'],
            'model_id': model_id,
            'model_type': first['model_type'],
            'hidden_dim': hidden_dim,
            'mode': mode,
            'n_runs': len(group),
            
            'delta_nll_mean': round(delta_nll_mean, 6),
            'delta_nll_min': round(delta_nll_min, 6),
            'delta_nll_max': round(delta_nll_max, 6),
            'delta_nll_p_perm_mean': round(p_mean, 6),
            
            'probe_r2_trained_mean': round(r2_trained_mean, 6) if r2_trained_mean is not None else None,
            'probe_r2_trained_min': round(r2_trained_min, 6) if r2_trained_min is not None else None,
            'probe_r2_trained_max': round(r2_trained_max, 6) if r2_trained_max is not None else None,
            
            'probe_r2_random_mean': round(r2_random_mean, 6) if r2_random_mean is not None else None,
            'probe_r2_random_min': round(r2_random_min, 6) if r2_random_min is not None else None,
            'probe_r2_random_max': round(r2_random_max, 6) if r2_random_max is not None else None,
            
            'probe_r2_permuted_mean': round(r2_permuted_mean, 6) if r2_permuted_mean is not None else None,
            'probe_r2_permuted_min': round(r2_permuted_min, 6) if r2_permuted_min is not None else None,
            'probe_r2_permuted_max': round(r2_permuted_max, 6) if r2_permuted_max is not None else None,
        })
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument(
        '--summary',
        type=Path,
        default=Path('experiments/results/summary.jsonl'),
        help='Path to summary.jsonl'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('experiments/results/aggregated_results.csv'),
        help='Path to output CSV'
    )
    args = parser.parse_args()
    
    print(f"[AGG] Reading summary log from: {args.summary}")
    runs = load_summary(args.summary)
    print(f"[AGG] Parsed {len(runs)} run entries.")

    # Filter out invalid runs
    runs_before = len(runs)
    runs = [r for r in runs if r.get('valid', True)]  # Default to valid if field missing
    runs_after = len(runs)
    if runs_before > runs_after:
        print(f"[AGG] Filtered out {runs_before - runs_after} invalid runs ({runs_after} valid runs remain).")

    print(f"[AGG] Aggregating by (world_id, model_id, hidden_dim, mode)...")
    aggregated = aggregate_runs(runs)
    print(f"[AGG] Aggregated {len(aggregated)} grouped entries.")
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(aggregated)
    df = df.sort_values(['world_id', 'model_id', 'hidden_dim', 'mode'])
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[AGG] Writing aggregated results to: {args.output}")
    df.to_csv(args.output, index=False)
    
    print(f"[AGG] Done. Aggregated {len(aggregated)} grouped entries.")
    
    # Print preview
    print("\n[AGG] Preview of aggregated results:")
    print(df[['world_id', 'model_id', 'hidden_dim', 'mode', 'delta_nll_mean', 'probe_r2_trained_mean']].to_string())


if __name__ == '__main__':
    main()
