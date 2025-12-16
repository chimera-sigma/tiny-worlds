# Clean, high-powered experimental runs for the paper
# 20 seeds per condition, 9999 permutations
# Run this from the tiny_world_model directory

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Clean Experimental Runs for Paper" -ForegroundColor Cyan
Write-Host "20 seeds, 9999 permutations per condition" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# E3 Oscillator - RNN H=16 (Full, Readout-Only, Random, Scrambled)
Write-Host "`n[E3 RNN H=16] Running 4 conditions..." -ForegroundColor Yellow
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 train=readout_only experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 train.max_steps=0 experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=rnn_tanh model.hidden_dim=16 train=scrambled experiment.n_seeds=20 stats.n_permutations=9999

# E3 Oscillator - RNN variants (Null, Nonstat)
Write-Host "`n[E3 RNN variants] Running null and nonstat..." -ForegroundColor Yellow
python run_experiment.py world=E3_osc_null model=rnn_tanh model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_nonstat model=rnn_tanh model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999

# E3 Oscillator - GRU H=16 (Full, Readout-Only, Random, Scrambled)
Write-Host "`n[E3 GRU H=16] Running 4 conditions..." -ForegroundColor Yellow
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 train=readout_only experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 train.max_steps=0 experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_struct model=gru model.hidden_dim=16 train=scrambled experiment.n_seeds=20 stats.n_permutations=9999

# E3 Oscillator - GRU variants (Null, Nonstat)
Write-Host "`n[E3 GRU variants] Running null and nonstat..." -ForegroundColor Yellow
python run_experiment.py world=E3_osc_null model=gru model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E3_osc_nonstat model=gru model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999

# E2 Regime - RNN (Struct Full, Struct Random, Null)
Write-Host "`n[E2 RNN] Running structured and null..." -ForegroundColor Yellow
python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 train.max_steps=0 experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E2_regime_null model=rnn_tanh model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999

# Optional: E2 RNN readout-only
# python run_experiment.py world=E2_regime_struct model=rnn_tanh model.hidden_dim=16 train=readout_only experiment.n_seeds=20 stats.n_permutations=9999

# E2 Regime - GRU (Struct Full, Null)
Write-Host "`n[E2 GRU] Running structured and null..." -ForegroundColor Yellow
python run_experiment.py world=E2_regime_struct model=gru model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999
python run_experiment.py world=E2_regime_null model=gru model.hidden_dim=16 train=default experiment.n_seeds=20 stats.n_permutations=9999

# Optional: E2 GRU readout-only
# python run_experiment.py world=E2_regime_struct model=gru model.hidden_dim=16 train=readout_only experiment.n_seeds=20 stats.n_permutations=9999

# ============================================
# CAPACITY SWEEPS (E3 only)
# ============================================

Write-Host "`n[Capacity Sweep] RNN H=2,4,8,16,32..." -ForegroundColor Yellow
python run_experiment.py -m world=E3_osc_struct model=rnn_tanh train=default experiment.n_seeds=20 stats.n_permutations=9999 model.hidden_dim=2,4,8,16,32

Write-Host "`n[Capacity Sweep] GRU H=2,4,8,16,32..." -ForegroundColor Yellow
python run_experiment.py -m world=E3_osc_struct model=gru train=default experiment.n_seeds=20 stats.n_permutations=9999 model.hidden_dim=2,4,8,16,32

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "All clean experiments completed!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Run: python scripts/aggregate_results.py" -ForegroundColor White
Write-Host "2. Run: python scripts/analyze_results.py" -ForegroundColor White
Write-Host "3. Compile paper: cd paper; .\compile.sh" -ForegroundColor White
