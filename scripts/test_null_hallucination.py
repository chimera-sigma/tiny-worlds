#!/usr/bin/env python3
"""
Structural Hallucination Test: Does the GRU dream structure in noise?

If GRU is a "High-Q Resonator" by design, it might impose order on chaos.
If RNN is a "Low-Q Damper", it should correctly identify null has no future.

This tests on NULL world (pure noise) to check for hallucination.

Usage:
    python scripts/test_null_hallucination.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
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


def compute_lyapunov_spectrum(model, x_seq, device, T_warmup=20, n_steps=100):
    """Compute Lyapunov spectrum (simplified from main script)."""
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


def test_coasting(model, x_seq, device, coast_duration=50, start_step=50):
    """Test coasting behavior - does it hallucinate structure?"""
    model.eval()
    H = model.hidden_dim
    seq_len = x_seq.shape[1]

    predictions = []
    hidden_norms = []

    with torch.no_grad():
        h = torch.zeros(1, H, device=device)
        for t in range(seq_len):
            x_t = x_seq[:, t, :]

            # Cut input during coast window
            if start_step <= t < start_step + coast_duration:
                x_t = torch.zeros_like(x_t)

            h = model.step(h, x_t)
            pred = model.head(h)
            predictions.append(pred.cpu().numpy())
            hidden_norms.append(torch.norm(h).item())

    return np.array(predictions).squeeze(), np.array(hidden_norms)


def analyze_null_dynamics(model, world, cfg, device, model_name):
    """Full analysis of null-trained model."""
    print(f"\n{'='*50}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*50}")

    # Get eval data
    eval_loader = world.eval_dataloader(batch_size=1)
    x_in, y, z_lat = next(iter(eval_loader))
    x_in = x_in.to(device)
    x_tc = apply_time_cond(x_in, cfg.time_cond, world.T)

    # 1. Lyapunov Spectrum
    print("\n[Computing Lyapunov Spectrum...]")
    spectrum = compute_lyapunov_spectrum(model, x_tc, device, T_warmup=20, n_steps=100)

    l1 = spectrum[0]
    l_sum = np.sum(spectrum)
    print(f"  λ1 (leading): {l1:.4f}")
    print(f"  λ_sum (log det): {l_sum:.4f}")
    print(f"  λ2..N mean: {spectrum[1:].mean():.4f}")

    # 2. Coasting Test
    print("\n[Running Coasting/Hallucination Test...]")
    coast_duration = 50
    start_step = 50

    predictions, hidden_norms = test_coasting(model, x_tc, device,
                                               coast_duration=coast_duration,
                                               start_step=start_step)

    # Analyze coasting behavior
    coast_segment = predictions[start_step:start_step+coast_duration]
    coast_var = np.var(coast_segment)
    coast_mean = np.mean(np.abs(coast_segment))

    # Check for oscillation during coasting
    if len(coast_segment) > 10:
        autocorr = np.corrcoef(coast_segment[:-1], coast_segment[1:])[0, 1]
    else:
        autocorr = 0

    print(f"  Coasting variance: {coast_var:.6f}")
    print(f"  Coasting mean |pred|: {coast_mean:.6f}")
    print(f"  Coasting autocorr: {autocorr:.4f}")

    # Verdict
    if coast_var > 0.01 or coast_mean > 0.05:
        verdict = "HALLUCINATION"
        verdict_desc = "Trying to impose structure on noise"
    else:
        verdict = "NIHILISM"
        verdict_desc = "Correctly predicts mean/zero"

    print(f"\n  >> VERDICT: {verdict}")
    print(f"     {verdict_desc}")

    return {
        'spectrum': spectrum,
        'l1': l1,
        'l_sum': l_sum,
        'predictions': predictions,
        'hidden_norms': hidden_norms,
        'coast_var': coast_var,
        'coast_mean': coast_mean,
        'autocorr': autocorr,
        'verdict': verdict,
        'input': x_tc[0].cpu().numpy(),
        'start_step': start_step,
        'coast_duration': coast_duration,
    }


def plot_comparison(results_gru, results_rnn, save_path=None):
    """Create comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, (results, name, color) in enumerate([
        (results_gru, 'GRU (Null-Trained)', 'blue'),
        (results_rnn, 'RNN (Null-Trained)', 'red')
    ]):
        start = results['start_step']
        dur = results['coast_duration']
        T = len(results['predictions'])

        # Left: Predictions with coasting
        ax = axes[idx, 0]
        t = np.arange(T)
        ax.plot(t, results['input'][:, 0], 'k-', alpha=0.2, linewidth=0.5, label='Input (noise)')
        ax.plot(t, results['predictions'], color=color, linewidth=1.5, label='Prediction')
        ax.axvspan(start, start + dur, color='gray', alpha=0.2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{name}: Response to Null World', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Output')
        ax.set_ylim(-1.5, 1.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.text(0.02, 0.98, f"Verdict: {results['verdict']}",
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Middle: Zoomed coasting region
        ax = axes[idx, 1]
        coast_t = np.arange(start - 20, start + dur + 20)
        coast_pred = results['predictions'][start-20:start+dur+20]
        ax.plot(coast_t, coast_pred, color=color, linewidth=2)
        ax.axvspan(start, start + dur, color='gray', alpha=0.2, label='Input Cut')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{name}: Coasting Region (Zoomed)', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Output')
        ax.set_ylim(-0.5, 0.5)

        # Right: Lyapunov spectrum
        ax = axes[idx, 2]
        spectrum = results['spectrum']
        ax.bar(range(len(spectrum)), spectrum, color=color, alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(-0.1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{name}: Lyapunov Spectrum', fontweight='bold')
        ax.set_xlabel('Exponent Index')
        ax.set_ylabel('λ')
        ax.text(0.98, 0.98, f"λ1 = {results['l1']:.3f}",
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to {save_path}")

    plt.show()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("="*60)
    print("STRUCTURAL HALLUCINATION TEST")
    print("Testing on NULL WORLD (pure noise)")
    print("="*60)

    device = torch.device("cpu")
    set_all_seeds(cfg.random.master_seed)

    # Override to null world
    null_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    null_cfg['id'] = 'E3_osc_null'
    null_cfg['structured'] = False
    null_world_cfg = OmegaConf.create(null_cfg)

    print(f"\nWorld: E3_osc_null (pure noise)")
    print(f"Hidden dim: {cfg.model.hidden_dim}")

    # Build null world
    world = OscillatorWorld(null_world_cfg, device=device)

    input_dim = cfg.world.input_dim
    hidden_dim = cfg.model.hidden_dim

    # Train GRU on null world
    print(f"\n[Training GRU on NULL world...]")
    gru = create_model("gru", input_dim, hidden_dim, cfg.model.init, device)
    train_model(gru, world, cfg, device, max_steps=5000)

    # Train RNN on null world
    print(f"\n[Training RNN on NULL world...]")
    set_all_seeds(cfg.random.master_seed)
    rnn = create_model("rnn_tanh", input_dim, hidden_dim, cfg.model.init, device)
    train_model(rnn, world, cfg, device, max_steps=5000)

    # Analyze both
    results_gru = analyze_null_dynamics(gru, world, cfg, device, "NULL-TRAINED GRU")
    results_rnn = analyze_null_dynamics(rnn, world, cfg, device, "NULL-TRAINED RNN")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: HALLUCINATION TEST")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25} {'GRU':>15} {'RNN':>15}")
    print(f"{'-'*55}")
    print(f"{'λ1 (leading)':<25} {results_gru['l1']:>15.4f} {results_rnn['l1']:>15.4f}")
    print(f"{'λ_sum (log det)':<25} {results_gru['l_sum']:>15.4f} {results_rnn['l_sum']:>15.4f}")
    print(f"{'Coast variance':<25} {results_gru['coast_var']:>15.6f} {results_rnn['coast_var']:>15.6f}")
    print(f"{'Coast mean |pred|':<25} {results_gru['coast_mean']:>15.6f} {results_rnn['coast_mean']:>15.6f}")
    print(f"{'Verdict':<25} {results_gru['verdict']:>15} {results_rnn['verdict']:>15}")

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    if results_gru['verdict'] == 'HALLUCINATION':
        print("""
GRU: SCENARIO A - "The Dreamer"
  The GRU has strong inductive bias toward oscillation.
  It forces structure onto noise - it HALLUCINATES an oscillator.
  >> The GRU's superior performance on structured worlds comes at
     the cost of hallucinating structure in random noise.
""")
    else:
        print("""
GRU: SCENARIO B - "The Learner"
  The GRU successfully suppressed its resonance bias.
  It correctly identified null world has no predictable structure.
  >> The GRU has BOTH capacity for resonance (structured world)
     AND plasticity to shut it off (null world). Impressive!
""")

    if results_rnn['verdict'] == 'NIHILISM':
        print("""
RNN: "The Realist"
  The RNN correctly identified there is no signal.
  It collapsed to predicting the mean (zero).
  >> This is the expected behavior for a low-Q damped system.
""")
    else:
        print("""
RNN: Unexpected behavior - showing activity on null world.
  This suggests the RNN may have found spurious correlations.
""")

    # Save figure
    fig_path = Path(__file__).parent.parent / "experiments" / "results" / "figures" / "null_hallucination_test.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(results_gru, results_rnn, save_path=str(fig_path))


if __name__ == "__main__":
    main()
