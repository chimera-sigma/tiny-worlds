#!/usr/bin/env python3
"""
Analysis and plotting script for tiny_world_model experiments.
Generates tables and figures from summary.jsonl.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
SUMMARY_PATH = Path("experiments/results/summary.jsonl")
OUTPUT_DIR = Path("experiments/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_summary_jsonl(path):
    """Load summary.jsonl and return as list of dicts."""
    runs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                run = json.loads(line)
                # Convert timestamp to datetime for sorting
                run['timestamp_dt'] = datetime.fromisoformat(run['timestamp'])
                runs.append(run)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[WARN] Failed to parse line: {e}")
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


def get_latest_runs(runs):
    """
    For each (world_id, model_id, hidden_dim, mode) combination,
    keep only the latest run by timestamp.
    """
    # Group by key
    groups = {}
    for run in runs:
        mode = derive_mode(run)
        key = (
            run['world_id'],
            run['model_id'],
            run['hidden_dim'],
            mode
        )
        if key not in groups or run['timestamp_dt'] > groups[key]['timestamp_dt']:
            groups[key] = run
            groups[key]['mode'] = mode  # Store mode for later use

    return list(groups.values())

def safe_format(val, decimals=4):
    """Format value, handling NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"

# ============================================================================
# LOAD DATA
# ============================================================================
print("[ANALYSIS] Loading summary.jsonl...")
all_runs = load_summary_jsonl(SUMMARY_PATH)
print(f"[ANALYSIS] Loaded {len(all_runs)} total runs")

print("[ANALYSIS] Keeping only latest runs per configuration...")
runs = get_latest_runs(all_runs)
print(f"[ANALYSIS] {len(runs)} unique configurations")

# Convert to DataFrame for easier manipulation
# Mode is already computed in get_latest_runs
df = pd.DataFrame(runs)

print("\n" + "="*80)
print("TABLES")
print("="*80 + "\n")

# ============================================================================
# TABLE 1: E3 Oscillator (RNN) - Capacity vs Emergence
# ============================================================================
print("### Table 1: E3 Oscillator (Structured, Stationary) - RNN Capacity Sweep\n")

t1_df = df[
    (df['world_id'] == 'E3_osc_struct') &
    (df['model_id'] == 'rnn_tanh') &
    (df['mode'] == 'full')
].sort_values('hidden_dim')

if len(t1_df) > 0:
    print("| hidden_dim | ΔNLL mean | p_perm | Probe R² (trained) |")
    print("|------------|-----------|--------|-------------------|")
    for _, row in t1_df.iterrows():
        print(f"| {row['hidden_dim']:>10} | {safe_format(row['delta_nll_mean'])} | {safe_format(row['delta_nll_p_perm'])} | {safe_format(row['probe_r2_trained_mean'])} |")
else:
    print("*No data found for this configuration*")

print()

# ============================================================================
# TABLE 2: E3 Oscillator (GRU) - Capacity vs Emergence
# ============================================================================
print("### Table 2: E3 Oscillator (Structured, Stationary) - GRU Capacity Sweep\n")

t2_df = df[
    (df['world_id'] == 'E3_osc_struct') &
    (df['model_id'] == 'gru') &
    (df['mode'] == 'full')
].sort_values('hidden_dim')

if len(t2_df) > 0:
    print("| hidden_dim | ΔNLL mean | p_perm | Probe R² (trained) |")
    print("|------------|-----------|--------|-------------------|")
    for _, row in t2_df.iterrows():
        print(f"| {row['hidden_dim']:>10} | {safe_format(row['delta_nll_mean'])} | {safe_format(row['delta_nll_p_perm'])} | {safe_format(row['probe_r2_trained_mean'])} |")
else:
    print("*No data found for this configuration*")

print()

# ============================================================================
# TABLE 3: E3 Oscillator - Full vs Readout-Only vs Random
# ============================================================================
print("### Table 3: E3 Oscillator (RNN-16 and GRU-16) - Full vs Readout-Only vs Random\n")

t3_df = df[
    (df['world_id'] == 'E3_osc_struct') &
    (df['hidden_dim'] == 16)
].sort_values(['model_id', 'mode'])

if len(t3_df) > 0:
    print("| model_id | mode | ΔNLL mean | p_perm | Probe R² (trained) |")
    print("|----------|------|-----------|--------|-------------------|")
    for _, row in t3_df.iterrows():
        print(f"| {row['model_id']:>8} | {row['mode']:>12} | {safe_format(row['delta_nll_mean'], 2):>9} | {safe_format(row['delta_nll_p_perm'])} | {safe_format(row['probe_r2_trained_mean'])} |")
else:
    print("*No data found for this configuration*")

print()

# ============================================================================
# TABLE 4: E2 Regime (Structured) - RNN vs GRU
# ============================================================================
print("### Table 4: E2 Regime World (Structured) - RNN vs GRU\n")

t4_df = df[
    (df['world_id'] == 'E2_regime_struct') &
    (df['mode'] == 'full')
].sort_values('model_id')

if len(t4_df) > 0:
    print("| model_id | ΔNLL mean | p_perm | Probe R² (trained) |")
    print("|----------|-----------|--------|-------------------|")
    for _, row in t4_df.iterrows():
        print(f"| {row['model_id']:>8} | {safe_format(row['delta_nll_mean'], 2):>9} | {safe_format(row['delta_nll_p_perm'])} | {safe_format(row['probe_r2_trained_mean'])} |")
else:
    print("*No data found for this configuration*")

print()

# ============================================================================
# TABLE 5: E2 Regime (Null) - RNN vs GRU
# ============================================================================
print("### Table 5: E2 Regime World (Null) - Null Behavior\n")

t5_df = df[
    (df['world_id'] == 'E2_regime_null')
].sort_values('model_id')

if len(t5_df) > 0:
    print("| model_id | ΔNLL mean | p_perm | Probe R² (trained) |")
    print("|----------|-----------|--------|-------------------|")
    for _, row in t5_df.iterrows():
        print(f"| {row['model_id']:>8} | {safe_format(row['delta_nll_mean'])} | {safe_format(row['delta_nll_p_perm'])} | {safe_format(row['probe_r2_trained_mean'])} |")
else:
    print("*No data found for this configuration*")

print()

print("\n" + "="*80)
print("FIGURES")
print("="*80 + "\n")

# ============================================================================
# FIGURE 1: Capacity vs ΔNLL (RNN vs GRU)
# ============================================================================
print("[FIGURE 1] Generating capacity sweep plot...")

fig1_rnn = df[
    (df['world_id'] == 'E3_osc_struct') &
    (df['model_id'] == 'rnn_tanh') &
    (df['mode'] == 'full')
].sort_values('hidden_dim')

fig1_gru = df[
    (df['world_id'] == 'E3_osc_struct') &
    (df['model_id'] == 'gru') &
    (df['mode'] == 'full')
].sort_values('hidden_dim')

if len(fig1_rnn) > 0 or len(fig1_gru) > 0:
    plt.figure(figsize=(8, 6))
    
    if len(fig1_rnn) > 0:
        plt.plot(fig1_rnn['hidden_dim'], fig1_rnn['delta_nll_mean'], 
                marker='o', linewidth=2, markersize=8, label='RNN (tanh)')
    
    if len(fig1_gru) > 0:
        plt.plot(fig1_gru['hidden_dim'], fig1_gru['delta_nll_mean'], 
                marker='s', linewidth=2, markersize=8, label='GRU')
    
    plt.xlabel('Hidden Dimension', fontsize=12)
    plt.ylabel('ΔNLL (mean)', fontsize=12)
    plt.title('E3 Oscillator: Capacity vs Emergence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig1_path = OUTPUT_DIR / "fig_e3_capacity_rnn_gru.png"
    plt.savefig(fig1_path, dpi=150)
    print(f"  → Saved to: {fig1_path}")
    print(f"  → Shows: ΔNLL vs hidden_dim for RNN and GRU on E3_osc_struct")
    plt.close()
else:
    print("  → Skipped (no data)")

print()

# ============================================================================
# FIGURE 2: Geometry vs Dynamics (E3 variants)
# ============================================================================
print("[FIGURE 2] Generating geometry vs dynamics plot...")

conditions = ['E3_osc_struct', 'E3_osc_nonstat', 'E3_osc_null']
fig2_data = []

for world_id in conditions:
    row = df[
        (df['world_id'] == world_id) &
        (df['model_id'] == 'rnn_tanh') &
        (df['hidden_dim'] == 16) &
        (df['mode'] == 'full')
    ]
    if len(row) > 0:
        row = row.iloc[0]
        fig2_data.append({
            'world_id': world_id,
            'delta_nll': row['delta_nll_mean'],
            'probe_r2_trained': row['probe_r2_trained_mean'],
            'probe_r2_random': row['probe_r2_random_mean']
        })

if len(fig2_data) > 0:
    fig2_df = pd.DataFrame(fig2_data)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(fig2_df))
    width = 0.35
    
    # Bars for probe R²
    ax1.bar(x_pos - width/2, fig2_df['probe_r2_random'].fillna(0), 
           width, label='Probe R² (random)', alpha=0.7, color='lightgray')
    ax1.bar(x_pos + width/2, fig2_df['probe_r2_trained'].fillna(0), 
           width, label='Probe R² (trained)', alpha=0.9, color='steelblue')
    
    ax1.set_ylabel('Probe R²', fontsize=12)
    ax1.set_ylim([0, 1.1])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([w.replace('E3_osc_', '') for w in fig2_df['world_id']], fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Line for ΔNLL
    ax2 = ax1.twinx()
    ax2.plot(x_pos, fig2_df['delta_nll'], 
            marker='D', linewidth=2.5, markersize=10, 
            color='darkred', label='ΔNLL mean')
    ax2.set_ylabel('ΔNLL (mean)', fontsize=12, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.title('E3 Oscillator: Geometry vs Dynamics (RNN-16)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig2_path = OUTPUT_DIR / "fig_e3_geometry_vs_dynamics_rnn.png"
    plt.savefig(fig2_path, dpi=150)
    print(f"  → Saved to: {fig2_path}")
    print(f"  → Shows: Probe R² and ΔNLL across E3 variants (struct, nonstat, null)")
    plt.close()
else:
    print("  → Skipped (no data)")

print()

# ============================================================================
# FIGURE 3: E2 Regime Decoding
# ============================================================================
print("[FIGURE 3] Generating E2 regime decoding plot...")

fig3_configs = [
    ('E2_regime_struct', 'rnn_tanh', 'E2_struct_RNN'),
    ('E2_regime_struct', 'gru', 'E2_struct_GRU'),
    ('E2_regime_null', 'rnn_tanh', 'E2_null_RNN'),
    ('E2_regime_null', 'gru', 'E2_null_GRU'),
]

fig3_data = []
for world_id, model_id, label in fig3_configs:
    row = df[
        (df['world_id'] == world_id) &
        (df['model_id'] == model_id)
    ]
    if len(row) > 0:
        row = row.iloc[0]
        fig3_data.append({
            'label': label,
            'probe_r2_trained': row['probe_r2_trained_mean'],
            'delta_nll': row['delta_nll_mean']
        })

if len(fig3_data) > 0:
    fig3_df = pd.DataFrame(fig3_data)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(fig3_df))
    
    # Bars for probe R² (main metric)
    colors = ['steelblue', 'lightcoral', 'gray', 'lightgray']
    bars = ax1.bar(x_pos, fig3_df['probe_r2_trained'].fillna(0), 
                   color=colors[:len(fig3_df)], alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('Probe R² (regime decoding)', fontsize=12)
    ax1.set_ylim([0, 1.0])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(fig3_df['label'], fontsize=11, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Overlay ΔNLL as markers
    ax2 = ax1.twinx()
    ax2.plot(x_pos, fig3_df['delta_nll'], 
            marker='o', linewidth=0, markersize=12, 
            color='darkgreen', label='ΔNLL', markeredgecolor='black', markeredgewidth=1.5)
    ax2.set_ylabel('ΔNLL (mean)', fontsize=12, color='darkgreen')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.title('E2 Regime: Decoding Performance (Structured vs Null, RNN vs GRU)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig3_path = OUTPUT_DIR / "fig_e2_regime_decoding.png"
    plt.savefig(fig3_path, dpi=150)
    print(f"  → Saved to: {fig3_path}")
    print(f"  → Shows: Probe R² for regime decoding across E2 worlds and architectures")
    plt.close()
else:
    print("  → Skipped (no data)")

print()
print("[ANALYSIS] Complete!")
print(f"[ANALYSIS] Figures saved to: {OUTPUT_DIR}")
