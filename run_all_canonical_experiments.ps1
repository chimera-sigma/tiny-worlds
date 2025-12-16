# ============================================================
# Tiny Emergence – Full Canonical Experiment Suite (20 seeds)
# ============================================================

# 0. Environment setup
cd D:\tiny_world_model
conda activate tiny_world_model

# 1. Archive old logs (safe reset)
mkdir .\experiments\results\archive 2>$null

if (Test-Path .\experiments\results\summary.jsonl) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    move .\experiments\results\summary.jsonl `
         .\experiments\results\archive\summary_OLD_$ts.jsonl 2>$null
}
if (Test-Path .\experiments\results\aggregated_results.csv) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    move .\experiments\results\aggregated_results.csv `
         .\experiments\results\archive\aggregated_results_OLD_$ts.csv 2>$null
}

# ============================================================
# 2. E3 Oscillator – RNN (tanh), H = 16
# ============================================================

# Structured world, full training
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# Structured world, readout-only (frozen core, train head)
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 `
    train=readout_only experiment.n_seeds=20 stats.n_permutations=9999

# Structured world, random core (no training)
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 `
    train.max_steps=0 experiment.n_seeds=20 stats.n_permutations=9999

# Structured world, scrambled core (full train + scramble before eval)
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 `
    train=scrambled experiment.n_seeds=20 stats.n_permutations=9999

# Null oscillator, full training
python run_experiment.py world=E3_osc_null model=rnn_tanh model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# Non-stationary oscillator, full training
python run_experiment.py world=E3_osc_nonstat model=rnn_tanh model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# ============================================================
# 3. E3 Oscillator – GRU, H = 16
# ============================================================

# Structured world, full training
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# Structured world, readout-only
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 `
    train=readout_only experiment.n_seeds=20 stats.n_permutations=9999

# Structured world, random core (no training)
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 `
    train.max_steps=0 experiment.n_seeds=20 stats.n_permutations=9999

# Structured world, scrambled core
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 `
    train=scrambled experiment.n_seeds=20 stats.n_permutations=9999

# Null oscillator, full training
python run_experiment.py world=E3_osc_null model=gru model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# Non-stationary oscillator, full training
python run_experiment.py world=E3_osc_nonstat model=gru model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# ============================================================
# 4. E2 Hidden Regime – RNN + GRU, H = 16
# ============================================================

# E2 structured, RNN full
python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# E2 structured, RNN random
python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 `
    train.max_steps=0 experiment.n_seeds=20 stats.n_permutations=9999

# E2 structured, RNN readout-only (optional, for completeness)
python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 `
    train=readout_only experiment.n_seeds=20 stats.n_permutations=9999

# E2 structured, GRU full
python run_experiment.py world=E2_regime_struct model=gru model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# E2 structured, GRU readout-only (optional)
python run_experiment.py world=E2_regime_struct model=gru model.hidden_dim=16 `
    train=readout_only experiment.n_seeds=20 stats.n_permutations=9999

# E2 null, RNN full
python run_experiment.py world=E2_regime_null model=rnn_tanh model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# E2 null, GRU full
python run_experiment.py world=E2_regime_null model=gru model.hidden_dim=16 `
    train=default experiment.n_seeds=20 stats.n_permutations=9999

# ============================================================
# 5. E3 Capacity Sweeps – RNN and GRU
# ============================================================

# RNN capacity sweep on E3 (H = 2,4,8,16,32)
python run_experiment.py -m world=E3_osc_struct model=rnn_tanh train=default `
    experiment.n_seeds=20 stats.n_permutations=9999 model.hidden_dim=2,4,8,16,32

# GRU capacity sweep on E3 (H = 2,4,8,16,32)
python run_experiment.py -m world=E3_osc_struct model=gru train=default `
    experiment.n_seeds=20 stats.n_permutations=9999 model.hidden_dim=2,4,8,16,32

# ============================================================
# 6. OPTIONAL: Time channel ablations (E3 and E2, RNN + GRU, H = 16)
# ============================================================

# E3 oscillator, RNN, time configs
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 `
    time_cond=full_time experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 `
    time_cond=no_time experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 `
    time_cond=constant_time experiment.n_seeds=20 stats.n_permutations=9999

# E3 oscillator, GRU, time configs
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 `
    time_cond=full_time experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 `
    time_cond=no_time experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 `
    time_cond=constant_time experiment.n_seeds=20 stats.n_permutations=9999

# E2 regime, RNN, time configs
python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 `
    time_cond=full_time experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 `
    time_cond=no_time experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 `
    time_cond=constant_time experiment.n_seeds=20 stats.n_permutations=9999

# E2 regime, GRU, time configs
python run_experiment.py world=E2_regime_struct model=gru model.hidden_dim=16 `
    time_cond=full_time experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E2_regime_struct model=gru model.hidden_dim=16 `
    time_cond=no_time experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E2_regime_struct model=gru model.hidden_dim=16 `
    time_cond=constant_time experiment.n_seeds=20 stats.n_permutations=9999

# ============================================================
# 7. Aggregate all results into aggregated_results.csv
# ============================================================

python scripts\aggregate_results.py
