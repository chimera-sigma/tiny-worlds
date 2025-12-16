#!/usr/bin/env python3
"""
Generate all paper figures from aggregated results.

Figures:
1. Capacity curves (E3, RNN vs GRU) - ΔNLL vs hidden_dim
2. Geometry vs Dynamics (modes at H=16) - full/readout/random/scrambled
3. Probe R² vs Capacity (E3) - shows geometry saturates quickly
4. E2 Emergence (discrete latent) - struct vs null
5. Time ablation summary - shows time channel is optional

Usage:
    python scripts/generate_paper_figures.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
SUMMARY_PATH = Path("experiments/results/summary.jsonl")
AGG_PATH = Path("experiments/results/aggregated_results.csv")
OUTPUT_DIR = Path("experiments/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Colors
RNN_COLOR = '#D62728'  # red
GRU_COLOR = '#1F77B4'  # blue
STRUCT_COLOR = '#2CA02C'  # green
NULL_COLOR = '#7F7F7F'  # gray


def load_data():
    """Load both summary.jsonl and aggregated CSV."""
    # Load summary for detailed per-run data
    runs = []
    with open(SUMMARY_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    summary_df = pd.DataFrame(runs)

    # Derive mode column
    def derive_mode(row):
        if row.get('readout_only', False):
            return 'readout_only'
        elif row.get('scramble_core', False):
            return 'scrambled'
        else:
            return 'full'

    summary_df['mode'] = summary_df.apply(derive_mode, axis=1)

    # Load aggregated
    agg_df = pd.read_csv(AGG_PATH)

    return summary_df, agg_df


def figure1_capacity_curves(df):
    """
    Figure 1: Capacity curves (E3, RNN vs GRU)
    X-axis: hidden_dim, Y-axis: ΔNLL
    """
    print("[Figure 1] Generating capacity curves...")

    # Filter for E3_osc_struct, full mode
    e3_full = df[
        (df['world_id'] == 'E3_osc_struct') &
        (df['mode'] == 'full')
    ]

    rnn_data = e3_full[e3_full['model_id'] == 'rnn_tanh'].sort_values('hidden_dim')
    gru_data = e3_full[e3_full['model_id'] == 'gru'].sort_values('hidden_dim')

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Plot GRU
    if len(gru_data) > 0:
        ax.plot(gru_data['hidden_dim'], gru_data['delta_nll_mean'],
                marker='o', markersize=9, linewidth=2.5, color=GRU_COLOR,
                label='GRU', zorder=3)

    # Plot RNN
    if len(rnn_data) > 0:
        ax.plot(rnn_data['hidden_dim'], rnn_data['delta_nll_mean'],
                marker='s', markersize=8, linewidth=2.5, color=RNN_COLOR,
                linestyle='--', label='RNN (tanh)', zorder=3)

    # Annotations
    if len(gru_data) > 0:
        h2_gru = gru_data[gru_data['hidden_dim'] == 2]
        h32_gru = gru_data[gru_data['hidden_dim'] == 32]
        if len(h2_gru) > 0 and len(h32_gru) > 0:
            h2_val = h2_gru['delta_nll_mean'].values[0]
            h32_val = h32_gru['delta_nll_mean'].values[0]
            pct = (h2_val / h32_val) * 100
            ax.annotate(f'{pct:.0f}% of H=32',
                       xy=(2, h2_val), xytext=(3.5, h2_val - 0.06),
                       fontsize=10, color=GRU_COLOR,
                       arrowprops=dict(arrowstyle='->', color=GRU_COLOR, lw=1.2))

    # Styling
    ax.set_xlabel('Hidden Dimension (H)', fontsize=12)
    ax.set_ylabel('ΔNLL (bits, higher = better)', fontsize=12)
    ax.set_title('E3 Oscillator: Capacity vs Emergence', fontsize=14, fontweight='bold')
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xscale('log', base=2)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_ylim(bottom=0.45)

    # Add horizontal line at approximate saturation
    ax.axhline(y=0.67, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(2.2, 0.675, 'saturation', fontsize=9, color='gray', alpha=0.7)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_e3_capacity_rnn_gru.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {out_path}")
    plt.close()


def figure2_geometry_vs_dynamics(df):
    """
    Figure 2: Geometry vs Dynamics (modes at H=16)
    Compare full/readout_only/scrambled for RNN and GRU
    """
    print("[Figure 2] Generating geometry vs dynamics comparison...")

    # Filter for E3_osc_struct, H=16
    e3_h16 = df[
        (df['world_id'] == 'E3_osc_struct') &
        (df['hidden_dim'] == 16)
    ]

    # Order: full, readout_only, scrambled
    mode_order = ['full', 'readout_only', 'scrambled']
    model_order = ['gru', 'rnn_tanh']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Prepare data
    x_labels = []
    dnll_vals = []
    probe_vals = []
    colors = []

    for model in model_order:
        model_label = 'GRU' if model == 'gru' else 'RNN'
        for mode in mode_order:
            row = e3_h16[(e3_h16['model_id'] == model) & (e3_h16['mode'] == mode)]
            if len(row) > 0:
                x_labels.append(f'{model_label}\n{mode.replace("_", " ")}')
                dnll_vals.append(row['delta_nll_mean'].values[0])
                probe_vals.append(row['probe_r2_trained_mean'].values[0] if pd.notna(row['probe_r2_trained_mean'].values[0]) else 0)
                colors.append(GRU_COLOR if model == 'gru' else RNN_COLOR)

    x_pos = np.arange(len(x_labels))

    # Panel 1: ΔNLL
    bars1 = ax1.bar(x_pos, dnll_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.set_ylabel('ΔNLL (bits)', fontsize=12)
    ax1.set_title('Predictive Performance (ΔNLL)', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, dnll_vals)):
        ypos = val + 0.02 if val >= 0 else val - 0.08
        ax1.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.2f}',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')

    # Panel 2: Probe R²
    bars2 = ax2.bar(x_pos, probe_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.set_ylabel('Probe R² (latent decoding)', fontsize=12)
    ax2.set_title('Geometric Representation (Probe R²)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars2, probe_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig2_e3_geometry_vs_dynamics.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {out_path}")
    plt.close()


def figure3_probe_capacity(df):
    """
    Figure 3: Probe R² vs Capacity (E3)
    Shows that geometry saturates even faster than ΔNLL
    """
    print("[Figure 3] Generating probe vs capacity...")

    e3_full = df[
        (df['world_id'] == 'E3_osc_struct') &
        (df['mode'] == 'full')
    ]

    rnn_data = e3_full[e3_full['model_id'] == 'rnn_tanh'].sort_values('hidden_dim')
    gru_data = e3_full[e3_full['model_id'] == 'gru'].sort_values('hidden_dim')

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot
    if len(gru_data) > 0:
        ax.plot(gru_data['hidden_dim'], gru_data['probe_r2_trained_mean'],
                marker='o', markersize=9, linewidth=2.5, color=GRU_COLOR,
                label='GRU')

    if len(rnn_data) > 0:
        ax.plot(rnn_data['hidden_dim'], rnn_data['probe_r2_trained_mean'],
                marker='s', markersize=8, linewidth=2.5, color=RNN_COLOR,
                linestyle='--', label='RNN (tanh)')

    ax.set_xlabel('Hidden Dimension (H)', fontsize=12)
    ax.set_ylabel('Probe R² (latent decoding)', fontsize=12)
    ax.set_title('E3 Oscillator: Geometric Representation vs Capacity', fontsize=14, fontweight='bold')
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xscale('log', base=2)
    ax.set_ylim(0.8, 1.01)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Note about RNN H=2
    if len(rnn_data) > 0:
        h2_rnn = rnn_data[rnn_data['hidden_dim'] == 2]
        if len(h2_rnn) > 0:
            r2_val = h2_rnn['probe_r2_trained_mean'].values[0]
            ax.annotate(f'RNN struggles\nat H=2',
                       xy=(2, r2_val), xytext=(3.2, r2_val - 0.05),
                       fontsize=9, color=RNN_COLOR,
                       arrowprops=dict(arrowstyle='->', color=RNN_COLOR, lw=1))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig3_e3_probe_capacity.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {out_path}")
    plt.close()


def figure4_e2_emergence(df):
    """
    Figure 4: E2 Emergence (discrete latent)
    Struct vs Null for RNN and GRU
    """
    print("[Figure 4] Generating E2 emergence comparison...")

    # Filter E2, H=16, full mode
    e2_data = df[
        (df['world_id'].str.startswith('E2_regime')) &
        (df['hidden_dim'] == 16) &
        (df['mode'] == 'full')
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Prepare data
    configs = [
        ('E2_regime_struct', 'gru', 'Struct-GRU'),
        ('E2_regime_struct', 'rnn_tanh', 'Struct-RNN'),
        ('E2_regime_null', 'gru', 'Null-GRU'),
        ('E2_regime_null', 'rnn_tanh', 'Null-RNN'),
    ]

    labels = []
    dnll_vals = []
    probe_vals = []
    colors = []

    for world_id, model_id, label in configs:
        row = e2_data[(e2_data['world_id'] == world_id) & (e2_data['model_id'] == model_id)]
        if len(row) > 0:
            labels.append(label)
            dnll_vals.append(row['delta_nll_mean'].values[0])
            probe_val = row['probe_r2_trained_mean'].values[0]
            probe_vals.append(probe_val if pd.notna(probe_val) else 0)
            if 'Struct' in label:
                colors.append(GRU_COLOR if 'GRU' in label else RNN_COLOR)
            else:
                colors.append('#AAAAAA')  # Gray for null

    x_pos = np.arange(len(labels))

    # Panel 1: ΔNLL
    bars1 = ax1.bar(x_pos, dnll_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax1.set_ylabel('ΔNLL (bits)', fontsize=12)
    ax1.set_title('E2 Regime World: Predictive Performance', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, dnll_vals):
        ypos = val + 0.3 if val >= 0 else val - 1
        ax1.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')

    # Panel 2: Probe R²
    bars2 = ax2.bar(x_pos, probe_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax2.set_ylabel('Probe R² (regime decoding)', fontsize=12)
    ax2.set_title('E2 Regime World: Representation Quality', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, probe_vals):
        label_text = f'{val:.2f}' if val > 0.01 else '—'
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, label_text,
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_e2_struct_vs_null.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {out_path}")
    plt.close()


def figure5_time_ablation(summary_df):
    """
    Figure 5: Time ablation summary
    Shows that explicit time channel is not required

    Note: This requires time_cond data which may not be in current results.
    If not available, we'll generate a placeholder.
    """
    print("[Figure 5] Checking for time ablation data...")

    # Check if time_cond column exists
    if 'time_cond' not in summary_df.columns:
        print("  → No time ablation data found. Skipping Figure 5.")
        print("  → To generate: run experiments with time_cond={full_time,no_time,constant_time}")
        return

    # Filter relevant data
    time_data = summary_df[
        (summary_df['world_id'].isin(['E3_osc_struct', 'E2_regime_struct'])) &
        (summary_df['hidden_dim'] == 16) &
        (summary_df['mode'] == 'full')
    ]

    if len(time_data) == 0 or time_data['time_cond'].nunique() < 2:
        print("  → Insufficient time ablation data. Skipping Figure 5.")
        return

    # Plot time ablation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    time_conds = ['full_time', 'no_time', 'constant_time']

    for idx, world_id in enumerate(['E3_osc_struct', 'E2_regime_struct']):
        ax = axes[idx]
        world_data = time_data[time_data['world_id'] == world_id]

        for model_id, color, marker in [('gru', GRU_COLOR, 'o'), ('rnn_tanh', RNN_COLOR, 's')]:
            model_data = world_data[world_data['model_id'] == model_id]
            if len(model_data) > 0:
                dnll_by_tc = []
                for tc in time_conds:
                    tc_data = model_data[model_data['time_cond'] == tc]
                    if len(tc_data) > 0:
                        dnll_by_tc.append(tc_data['delta_nll_mean'].mean())
                    else:
                        dnll_by_tc.append(np.nan)

                ax.plot(range(len(time_conds)), dnll_by_tc,
                       marker=marker, markersize=8, linewidth=2,
                       color=color, label=model_id.upper().replace('_', ' '))

        ax.set_xticks(range(len(time_conds)))
        ax.set_xticklabels([tc.replace('_', '\n') for tc in time_conds])
        ax.set_ylabel('ΔNLL (bits)')
        ax.set_title(f'{world_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Time Ablation: Explicit Time Not Required', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = OUTPUT_DIR / "fig5_time_ablation.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {out_path}")
    plt.close()


def main():
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    summary_df, agg_df = load_data()
    print(f"[DATA] Loaded {len(summary_df)} runs from summary.jsonl")
    print(f"[DATA] Loaded {len(agg_df)} aggregated entries")
    print()

    # Generate all figures
    figure1_capacity_curves(agg_df)
    figure2_geometry_vs_dynamics(agg_df)
    figure3_probe_capacity(agg_df)
    figure4_e2_emergence(agg_df)
    figure5_time_ablation(summary_df)

    print()
    print("=" * 60)
    print(f"COMPLETE. Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
