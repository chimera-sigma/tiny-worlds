# Paper Compilation Directory

This directory contains the LaTeX source for the paper "Emergence Without Scale: How Tiny Recurrent Networks Learn Dynamical Operators".

## Files

- `paper.tex` - Main LaTeX source with tables populated from experimental results
- `fig_e3_capacity_rnn_gru.png` - Capacity vs ΔNLL plot (RNN vs GRU)
- `fig_e3_geometry_vs_dynamics_rnn.png` - Geometry vs dynamics across E3 variants
- `fig_e2_regime_decoding.png` - E2 regime decoding performance

## Compiling

### Option 1: Local LaTeX installation

```bash
cd paper
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references
```

### Option 2: Overleaf

1. Zip the contents of this directory
2. Upload to Overleaf
3. Compile with pdflatex

## Generating Figures

Figures need to be generated from the experimental results. Install matplotlib first:

```bash
pip install matplotlib
```

Then run the analysis script:

```bash
cd /mnt/d/tiny_world_model
python3 scripts/analyze_results.py
```

This will create the three figures in `experiments/results/figures/` and you can copy them here:

```bash
cp experiments/results/figures/*.png paper/
```

## Table Values

All tables in the paper have been populated with exact values from the experimental runs logged in `experiments/results/summary.jsonl`. The values reflect the latest run for each configuration as of the compilation date.

### Table 1: E3 RNN Capacity Sweep
- Based on runs with world_id=E3_osc_struct, model_id=rnn_tanh, mode=full
- Hidden dimensions: 4, 8, 16, 32
- All runs significant at p ≈ 0.033

### Table 2: E3 GRU Capacity Sweep  
- Based on runs with world_id=E3_osc_struct, model_id=gru, mode=full
- Same hidden dimensions as RNN
- Shows GRU achieves slightly higher ΔNLL at all capacities

### Table 3: E3 Training Modes (H=16)
- Compares full training vs readout-only for RNN and GRU
- Shows GRU geometry provides some predictive power when frozen
- RNN readout-only fails to beat baseline

### Table 4: E2 Regime (Structured)
- Currently only GRU data available (RNN run missing from latest logs)
- Shows massive ΔNLL (≈16) for discrete latent regime task
- Regime decodability R² ≈ 0.81

## Missing Data

The current experimental log is missing:
- E2_regime_struct with RNN (full mode)
- E2_regime_null with GRU
- Random-weights control runs (they're in the data but filtered in analysis)

To complete the paper, rerun:
```bash
python3 run_experiment.py world=E2_regime_struct model=rnn_tanh
python3 run_experiment.py world=E2_regime_null model=gru
```
