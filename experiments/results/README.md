# Experiment Results

This directory contains raw experiment results. Results are excluded from git due to file size.

## How to Reproduce

```bash
# Run all experiments (requires GPU)
python scripts/run_experiments.py

# Or run individual experiments
python -m tiny_world_model.train world=drift model=gru
python -m tiny_world_model.train world=oscillator model=rnn
```

## Results Format

Each experiment produces:
- `metrics.csv`: Per-seed metrics (NLL, probe accuracy, etc.)
- `lyapunov.csv`: Lyapunov exponent estimates
- `checkpoints/`: Model weights (optional, if `save_checkpoints=true`)

## Obtaining Results

Full results are available on Zenodo: [DOI link will be added after publication]
