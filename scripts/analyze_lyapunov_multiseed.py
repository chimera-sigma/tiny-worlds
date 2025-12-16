#!/usr/bin/env python3
"""
Multi-seed Lyapunov Spectrum Analysis for statistical confidence intervals.

Runs Lyapunov analysis across multiple seeds to compute mean ± 95% CI.

Usage:
    python scripts/analyze_lyapunov_multiseed.py world=E3_osc_struct
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_world_model.worlds.oscillator import OscillatorWorld
from tiny_world_model.models.rnn import TinyRNN, TinyGRU
from tiny_world_model.time_cond import apply_time_cond
from tiny_world_model.utils import set_all_seeds


def create_model(model_type, input_dim, hidden_dim, init_cfg, device):
    if model_type == "rnn_tanh":
        return TinyRNN(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=init_cfg).to(device)
    elif model_type == "gru":
        return TinyGRU(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=init_cfg).to(device)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")


def train_model(model, world, cfg, device, max_steps=5000):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.optimizer.lr)
    train_loader = world.train_dataloader(cfg.train.batch_size)
    step = 0

    for epoch in range(9999):
        for x_in, y, z_lat in train_loader:
            x_in = x_in.to(device)
            y = y.to(device)
            x_tc = apply_time_cond(x_in, cfg.time_cond, world.T)

            pred, h_seq = model(x_tc)
            loss = F.mse_loss(pred[:, :-1], y[:, 1:])

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step >= max_steps:
                return
        if step >= max_steps:
            return


def compute_lyapunov_spectrum(model, x_seq, device, T_warmup=50, n_steps=150):
    """Compute Lyapunov spectrum using QR decomposition."""
    model.train()
    H = model.hidden_dim
    h = torch.zeros(1, H, device=device)
    Q = torch.eye(H, device=device)
    log_r_sums = np.zeros(H)
    valid_steps = 0

    T_total = min(T_warmup + n_steps, x_seq.shape[1] - 1)

    with torch.no_grad():
        for t in range(T_warmup):
            x_t = x_seq[:, t, :]
            h = model.step(h, x_t)

    for t in range(T_warmup, T_total):
        x_t = x_seq[:, t, :].detach()
        h_in = h.clone().detach().requires_grad_(True)
        h_next = model.step(h_in, x_t)

        jac_cols = []
        for i in range(H):
            grad_outputs = torch.zeros_like(h_next)
            grad_outputs[0, i] = 1.0
            grad = torch.autograd.grad(h_next, h_in, grad_outputs=grad_outputs,
                                       retain_graph=True, create_graph=False)[0]
            jac_cols.append(grad.view(-1))

        J = torch.stack(jac_cols, dim=0)
        M = torch.matmul(J, Q)
        Q_new, R = torch.linalg.qr(M)
        diag_R = torch.diag(R)
        log_diag = torch.log(torch.abs(diag_R) + 1e-12)
        log_r_sums += log_diag.detach().cpu().numpy()

        h = h_next.detach()
        Q = Q_new.detach()
        valid_steps += 1

    if valid_steps == 0:
        return np.zeros(H)

    lyap_exponents = log_r_sums / valid_steps
    return np.sort(lyap_exponents)[::-1]


def compute_ci(values, confidence=0.95):
    """Compute mean and 95% confidence interval."""
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, ci


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("="*60)
    print("MULTI-SEED LYAPUNOV SPECTRUM ANALYSIS")
    print("="*60)

    device = torch.device("cpu")
    n_seeds = 5
    hidden_dim = 16

    # Build world
    world = OscillatorWorld(cfg.world, device=device)
    input_dim = cfg.world.input_dim

    # Storage for results
    results = {
        'gru_trained': {'l1': [], 'l_rest_mean': [], 'l_sum': []},
        'gru_random': {'l1': [], 'l_rest_mean': [], 'l_sum': []},
        'rnn_trained': {'l1': [], 'l_rest_mean': [], 'l_sum': []},
        'rnn_random': {'l1': [], 'l_rest_mean': [], 'l_sum': []},
    }

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        set_all_seeds(cfg.random.master_seed + seed)

        # Get eval data
        eval_loader = world.eval_dataloader(batch_size=1)
        x_in, y, z_lat = next(iter(eval_loader))
        x_in = x_in.to(device)
        x_tc = apply_time_cond(x_in, cfg.time_cond, world.T)

        # GRU Random
        gru_random = create_model("gru", input_dim, hidden_dim, cfg.model.init, device)
        spectrum = compute_lyapunov_spectrum(gru_random, x_tc, device)
        results['gru_random']['l1'].append(spectrum[0])
        results['gru_random']['l_rest_mean'].append(spectrum[1:].mean())
        results['gru_random']['l_sum'].append(spectrum.sum())

        # GRU Trained
        gru_trained = create_model("gru", input_dim, hidden_dim, cfg.model.init, device)
        train_model(gru_trained, world, cfg, device, max_steps=5000)
        spectrum = compute_lyapunov_spectrum(gru_trained, x_tc, device)
        results['gru_trained']['l1'].append(spectrum[0])
        results['gru_trained']['l_rest_mean'].append(spectrum[1:].mean())
        results['gru_trained']['l_sum'].append(spectrum.sum())

        # RNN Random
        set_all_seeds(cfg.random.master_seed + seed)
        rnn_random = create_model("rnn_tanh", input_dim, hidden_dim, cfg.model.init, device)
        spectrum = compute_lyapunov_spectrum(rnn_random, x_tc, device)
        results['rnn_random']['l1'].append(spectrum[0])
        results['rnn_random']['l_rest_mean'].append(spectrum[1:].mean())
        results['rnn_random']['l_sum'].append(spectrum.sum())

        # RNN Trained
        rnn_trained = create_model("rnn_tanh", input_dim, hidden_dim, cfg.model.init, device)
        train_model(rnn_trained, world, cfg, device, max_steps=5000)
        spectrum = compute_lyapunov_spectrum(rnn_trained, x_tc, device)
        results['rnn_trained']['l1'].append(spectrum[0])
        results['rnn_trained']['l_rest_mean'].append(spectrum[1:].mean())
        results['rnn_trained']['l_sum'].append(spectrum.sum())

    # Print results table
    print(f"\n{'='*80}")
    print("LYAPUNOV SPECTRUM: MULTI-SEED RESULTS (mean ± 95% CI)")
    print(f"{'='*80}")
    print(f"\n{'Model':<12} {'Condition':<15} {'λ1':>20} {'λ>1 mean':>20} {'Σλ':>15}")
    print(f"{'-'*80}")

    for key, label in [('gru_trained', 'GRU Trained'),
                       ('gru_random', 'GRU Random'),
                       ('rnn_trained', 'RNN Trained'),
                       ('rnn_random', 'RNN Random')]:
        l1_mean, l1_ci = compute_ci(results[key]['l1'])
        lrest_mean, lrest_ci = compute_ci(results[key]['l_rest_mean'])
        lsum_mean, lsum_ci = compute_ci(results[key]['l_sum'])

        model = 'GRU' if 'gru' in key else 'RNN'
        cond = 'Trained (E3)' if 'trained' in key else 'Random'

        print(f"{model:<12} {cond:<15} {l1_mean:>8.3f} ± {l1_ci:.3f}   {lrest_mean:>8.2f} ± {lrest_ci:.2f}   {lsum_mean:>8.2f} ± {lsum_ci:.2f}")

    # LaTeX table format
    print(f"\n{'='*80}")
    print("LATEX TABLE FORMAT")
    print(f"{'='*80}")

    print("""
\\begin{tabular}{llcccc}
    \\toprule
    Model & Condition & $\\lambda_1$ & $\\lambda_{>1}$ mean & $\\sum \\lambda$ & Interpretation \\\\
    \\midrule""")

    for key, interp in [('gru_trained', 'High-Q Resonator'),
                        ('gru_random', 'Stable sink'),
                        ('rnn_trained', 'Low-Q Damped Oscillator'),
                        ('rnn_random', 'Weak sink')]:
        l1_mean, l1_ci = compute_ci(results[key]['l1'])
        lrest_mean, lrest_ci = compute_ci(results[key]['l_rest_mean'])
        lsum_mean, lsum_ci = compute_ci(results[key]['l_sum'])

        model = 'GRU' if 'gru' in key else 'RNN'
        cond = 'Trained (E3)' if 'trained' in key else 'Random'

        if 'gru' in key and 'trained' in key:
            print(f"    \\multirow{{2}}{{*}}{{GRU}}")
        if 'rnn' in key and 'trained' in key:
            print(f"    \\midrule")
            print(f"    \\multirow{{2}}{{*}}{{RNN}}")

        print(f"      & {cond}  & ${l1_mean:.3f} \\pm {l1_ci:.3f}$ & ${lrest_mean:.2f} \\pm {lrest_ci:.2f}$ & ${lsum_mean:.2f}$ & {interp} \\\\")

    print("""    \\bottomrule
  \\end{tabular}""")

    # Check statistical significance
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TEST")
    print(f"{'='*80}")

    gru_l1 = results['gru_trained']['l1']
    rnn_l1 = results['rnn_trained']['l1']

    t_stat, p_value = stats.ttest_ind(gru_l1, rnn_l1)
    print(f"\nTwo-sample t-test for λ1 (GRU trained vs RNN trained):")
    print(f"  GRU λ1: {np.mean(gru_l1):.4f} ± {np.std(gru_l1):.4f}")
    print(f"  RNN λ1: {np.mean(rnn_l1):.4f} ± {np.std(rnn_l1):.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("  >> SIGNIFICANT: GRU and RNN have different λ1 (p < 0.05)")
    else:
        print("  >> NOT SIGNIFICANT at p < 0.05")


if __name__ == "__main__":
    main()
