# Quick Start: Clean Experimental Runs

## TL;DR Workflow

```powershell
# 0. Archive old messy data (recommended)
.\reset_for_clean_runs.ps1

# 1. Run all clean experiments (~600 runs, several hours on GPU)
.\run_clean_experiments.ps1

# 2. Aggregate results (filters invalid runs automatically)
python scripts\aggregate_results.py

# 3. Generate figures
python scripts\analyze_results.py

# 4. Compile paper
cd paper
.\compile.sh
```

## What You Get

✅ **20 seeds per configuration**
✅ **9,999 permutation samples**
✅ **>80% statistical power for d > 0.9**
✅ **Automatic quality control**
✅ **Publication-ready Methods paragraph**

## The Scripts

### `reset_for_clean_runs.ps1`
- Archives old logs to `experiments/results/archive_TIMESTAMP/`
- Clears summary.jsonl and aggregated_results.csv
- **Safe**: All old data preserved

### `run_clean_experiments.ps1`
- Runs ~30 experimental configurations
- Each with 20 seeds and 9,999 permutations
- Covers all paper figures and tables
- **Time**: Several hours on GPU

### `aggregate_results.py`
- Filters out invalid runs automatically
- Creates clean aggregated_results.csv
- Reports filtering statistics

### `analyze_results.py`
- Generates all paper figures
- Saves to experiments/results/figures/
- Uses clean aggregated data

## Paper Integration

The file `paper/statistical_methods_paragraph.tex` contains publication-ready text:

```latex
\subsection{Statistical evaluation}

All experiments were conducted with 20 independent random seeds...
[etc.]
```

**Just copy this into your Methods section.**

## Key Features

### Automatic Validity Checking
Every run is automatically checked for:
- NaN or Inf values
- Exploded training (delta_nll < -100)
- Invalid runs flagged and filtered

### Current Data Quality
- **41 old runs**: 1 invalid (2.4%)
- **After filtering**: 40 valid runs remain

### Effect Sizes
Your observed effect sizes are **d > 2** (often hundreds of SDs), far above the detection threshold of d > 0.9.

## Troubleshooting

**Q: How long will this take?**
A: ~600 training runs × ~3-5 minutes each on GPU = several hours. Run overnight.

**Q: Can I run experiments in parallel?**
A: Yes! The capacity sweeps use Hydra's `-m` flag for multirun. You can also manually run different conditions in parallel terminals.

**Q: What if some runs fail?**
A: They'll be automatically flagged as invalid and filtered during aggregation. Check the validity messages in the log.

**Q: Can I add more experiments later?**
A: Yes! Just run them and re-run aggregate_results.py. The new runs will be appended to summary.jsonl.

## Documentation

See `CLEAN_EXPERIMENTS_SETUP.md` for complete details on:
- Code infrastructure
- Statistical justification
- Experimental design
- Files modified/created
