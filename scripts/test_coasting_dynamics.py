#!/usr/bin/env python3
"""
Dynamical Object Permanence Test: The "Coasting" Experiment

If the GRU has learned conformal symplectic flow (λ1 ≈ 0), the hidden state
should act like a flywheel - when input is cut, momentum keeps it spinning.

If the RNN is a forced damped oscillator (λ1 < 0), cutting input removes
the forcing function and damping instantly kills the dynamics.

Usage:
    python scripts/test_coasting_dynamics.py world=E3_osc_struct
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
    """Factory function for creating models."""
    if model_type == "rnn_tanh":
        return TinyRNN(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=init_cfg).to(device)
    elif model_type == "gru":
        return TinyGRU(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=init_cfg).to(device)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")


def train_model(model, world, cfg, device, max_steps=5000):
    """Train model to convergence."""
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
            if step % 1000 == 0:
                print(f"  step {step}, loss={loss.item():.6f}")
            if step >= max_steps:
                return
        if step >= max_steps:
            return


def test_coasting_dynamics(model, world, cfg, device, coast_duration=50, start_step=100):
    """
    Tests if the model's hidden state continues to oscillate when input is cut.

    Returns:
        dict with predictions, ground truth, and metrics
    """
    model.eval()

    # Get evaluation data
    eval_loader = world.eval_dataloader(batch_size=1)
    x_in, y, z_lat = next(iter(eval_loader))
    x_in = x_in.to(device)
    y = y.to(device)
    x_tc = apply_time_cond(x_in, cfg.time_cond, world.T)

    seq_len = x_tc.shape[1]
    H = model.hidden_dim

    # Storage
    predictions_normal = []
    predictions_coast = []
    hidden_states_coast = []

    # Run 1: Normal (no intervention) - baseline
    with torch.no_grad():
        h = torch.zeros(1, H, device=device)
        for t in range(seq_len):
            x_t = x_tc[:, t, :]
            h = model.step(h, x_t)
            pred = model.head(h)
            predictions_normal.append(pred.cpu().numpy())

    # Run 2: With coasting intervention
    with torch.no_grad():
        h = torch.zeros(1, H, device=device)
        for t in range(seq_len):
            x_t = x_tc[:, t, :]

            # THE INTERVENTION: Zero out input during coast window
            if start_step <= t < start_step + coast_duration:
                x_t = torch.zeros_like(x_t)

            h = model.step(h, x_t)
            pred = model.head(h)
            predictions_coast.append(pred.cpu().numpy())

            # Store hidden state during and around coast
            if start_step - 10 <= t < start_step + coast_duration + 10:
                hidden_states_coast.append(h.cpu().numpy())

    predictions_normal = np.concatenate(predictions_normal, axis=0)  # [T, d_out]
    predictions_coast = np.concatenate(predictions_coast, axis=0)
    ground_truth = y[0].cpu().numpy()  # [T, d_out]
    latent = z_lat[0].cpu().numpy()  # [T, d_lat]

    return {
        'predictions_normal': predictions_normal,
        'predictions_coast': predictions_coast,
        'ground_truth': ground_truth,
        'latent': latent,
        'hidden_states': np.array(hidden_states_coast),
        'start_step': start_step,
        'coast_duration': coast_duration,
    }


def compute_coasting_metrics(results, model_name):
    """Compute quantitative metrics for coasting behavior."""
    pred = results['predictions_coast']
    start = results['start_step']
    dur = results['coast_duration']

    # Amplitude before coasting (last 20 steps before intervention)
    pre_window = pred[start-20:start, 0]
    amp_before = np.std(pre_window)

    # Amplitude at end of coasting (last 20 steps of coast)
    coast_end_window = pred[start+dur-20:start+dur, 0]
    amp_after = np.std(coast_end_window)

    # Amplitude retention
    retention = amp_after / (amp_before + 1e-9)

    # Phase coherence: check if oscillation continues
    # Compute autocorrelation lag-1 during coast
    coast_segment = pred[start:start+dur, 0]
    if len(coast_segment) > 10:
        autocorr = np.corrcoef(coast_segment[:-1], coast_segment[1:])[0, 1]
    else:
        autocorr = 0

    # Mean drift from zero (collapse to mean?)
    mean_during_coast = np.mean(np.abs(coast_segment))

    return {
        'model': model_name,
        'amp_before': amp_before,
        'amp_after': amp_after,
        'retention': retention,
        'autocorr': autocorr,
        'mean_abs': mean_during_coast,
    }


def plot_coasting_comparison(results_gru, results_rnn, metrics_gru, metrics_rnn, save_path=None):
    """Create publication-quality comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors
    color_gt = '#888888'
    color_normal = '#2ecc71'
    color_coast = '#3498db'

    for idx, (results, metrics, name) in enumerate([
        (results_gru, metrics_gru, 'GRU'),
        (results_rnn, metrics_rnn, 'RNN')
    ]):
        ax_main = axes[idx, 0]
        ax_hidden = axes[idx, 1]

        start = results['start_step']
        dur = results['coast_duration']
        T = len(results['predictions_coast'])

        # Main plot: predictions
        t_range = np.arange(T)

        # Ground truth (latent)
        ax_main.plot(t_range, results['latent'][:, 0],
                    color=color_gt, linestyle='--', alpha=0.5,
                    linewidth=1, label='Latent (ground truth)')

        # Normal prediction (no intervention)
        ax_main.plot(t_range, results['predictions_normal'][:, 0],
                    color=color_normal, alpha=0.4, linewidth=1,
                    label='Prediction (normal)')

        # Coasting prediction
        ax_main.plot(t_range, results['predictions_coast'][:, 0],
                    color=color_coast, linewidth=2,
                    label='Prediction (with coast)')

        # Highlight coast zone
        ax_main.axvspan(start, start + dur, color='red', alpha=0.15)
        ax_main.axvline(start, color='red', linestyle=':', alpha=0.5)
        ax_main.axvline(start + dur, color='red', linestyle=':', alpha=0.5)

        # Add retention metric as text
        verdict = "INERTIAL" if metrics['retention'] > 0.5 else "DISSIPATIVE"
        ax_main.text(0.98, 0.95,
                    f"Amplitude Retention: {metrics['retention']:.1%}\nVerdict: {verdict}",
                    transform=ax_main.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_main.set_title(f'{name}: Coasting Dynamics Test', fontsize=12, fontweight='bold')
        ax_main.set_xlabel('Time Step')
        ax_main.set_ylabel('Output (dim 0)')
        ax_main.legend(loc='upper left', fontsize=9)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(start - 50, start + dur + 50)

        # Hidden state plot (if available)
        if len(results['hidden_states']) > 0:
            h_states = results['hidden_states'].squeeze()  # [T_window, H]
            # Plot first 3 hidden dimensions
            t_hidden = np.arange(start - 10, start + dur + 10)[:len(h_states)]
            for dim in range(min(3, h_states.shape[1])):
                ax_hidden.plot(t_hidden, h_states[:, dim],
                              linewidth=1.5, alpha=0.7,
                              label=f'h[{dim}]')

            ax_hidden.axvspan(start, start + dur, color='red', alpha=0.15)
            ax_hidden.set_title(f'{name}: Hidden State Dynamics', fontsize=12)
            ax_hidden.set_xlabel('Time Step')
            ax_hidden.set_ylabel('Hidden State Value')
            ax_hidden.legend(loc='upper right', fontsize=9)
            ax_hidden.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("="*60)
    print("DYNAMICAL OBJECT PERMANENCE: COASTING TEST")
    print("="*60)
    print(f"\nWorld: {cfg.world.id}")
    print(f"Hidden dim: {cfg.model.hidden_dim}")

    device = torch.device("cpu")
    set_all_seeds(cfg.random.master_seed)

    # Build world
    world = OscillatorWorld(cfg.world, device=device)

    input_dim = cfg.world.input_dim
    hidden_dim = cfg.model.hidden_dim

    # Coasting parameters
    coast_duration = 50  # Steps to cut input
    start_step = 100     # When to start coasting

    # Train and test GRU
    print(f"\n[Training GRU...]")
    gru = create_model("gru", input_dim, hidden_dim, cfg.model.init, device)
    train_model(gru, world, cfg, device, max_steps=5000)

    print(f"\n[Testing GRU coasting dynamics...]")
    results_gru = test_coasting_dynamics(gru, world, cfg, device,
                                          coast_duration=coast_duration,
                                          start_step=start_step)
    metrics_gru = compute_coasting_metrics(results_gru, "GRU")

    # Train and test RNN
    print(f"\n[Training RNN...]")
    set_all_seeds(cfg.random.master_seed)  # Reset seed for fair comparison
    rnn = create_model("rnn_tanh", input_dim, hidden_dim, cfg.model.init, device)
    train_model(rnn, world, cfg, device, max_steps=5000)

    print(f"\n[Testing RNN coasting dynamics...]")
    results_rnn = test_coasting_dynamics(rnn, world, cfg, device,
                                          coast_duration=coast_duration,
                                          start_step=start_step)
    metrics_rnn = compute_coasting_metrics(results_rnn, "RNN")

    # Print results
    print(f"\n{'='*60}")
    print("COASTING ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25} {'GRU':>12} {'RNN':>12}")
    print(f"{'-'*49}")
    print(f"{'Amplitude Before Coast':<25} {metrics_gru['amp_before']:>12.4f} {metrics_rnn['amp_before']:>12.4f}")
    print(f"{'Amplitude After Coast':<25} {metrics_gru['amp_after']:>12.4f} {metrics_rnn['amp_after']:>12.4f}")
    print(f"{'Retention Ratio':<25} {metrics_gru['retention']:>11.1%} {metrics_rnn['retention']:>11.1%}")
    print(f"{'Autocorrelation (coast)':<25} {metrics_gru['autocorr']:>12.4f} {metrics_rnn['autocorr']:>12.4f}")
    print(f"{'Mean |output| (coast)':<25} {metrics_gru['mean_abs']:>12.4f} {metrics_rnn['mean_abs']:>12.4f}")

    # Verdicts
    print(f"\n{'='*60}")
    print("VERDICTS")
    print(f"{'='*60}")

    gru_verdict = "INERTIAL (Hamiltonian-like)" if metrics_gru['retention'] > 0.5 else "DISSIPATIVE"
    rnn_verdict = "INERTIAL (Hamiltonian-like)" if metrics_rnn['retention'] > 0.5 else "DISSIPATIVE"

    print(f"\nGRU: {gru_verdict}")
    if metrics_gru['retention'] > 0.5:
        print("  >> Hidden state acts as flywheel - dynamics continue without input")
        print("  >> Model has internalized the oscillator as an autonomous system")
    else:
        print("  >> Hidden state collapses without input forcing")

    print(f"\nRNN: {rnn_verdict}")
    if metrics_rnn['retention'] > 0.5:
        print("  >> Hidden state acts as flywheel - dynamics continue without input")
    else:
        print("  >> Hidden state collapses without input forcing")
        print("  >> Model is a reflex agent: no input → no output")

    # Save figure
    fig_path = Path(__file__).parent.parent / "experiments" / "results" / "figures" / "coasting_dynamics.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plot_coasting_comparison(results_gru, results_rnn, metrics_gru, metrics_rnn,
                             save_path=str(fig_path))

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    if metrics_gru['retention'] > 0.5 and metrics_rnn['retention'] < 0.5:
        print("""
The results demonstrate a fundamental difference in learned dynamics:

GRU (λ1 ≈ 0): The gated architecture learns an AUTONOMOUS dynamical system.
  - When input is cut, the hidden state continues to oscillate
  - It has built an internal "physics engine" that generates the oscillator dynamics
  - This is conformal symplectic flow: geometry preserved, noise contracted

RNN (λ1 < 0): The vanilla architecture learns a FORCED dynamical system.
  - When input is cut, the hidden state decays to equilibrium
  - It is a sophisticated input→output mapping, not an internal simulator
  - This is pure dissipation: all directions contract including the flow

IMPLICATION FOR EMERGENCE:
  - GRU emergence = learning a dynamical LAW (autonomous system)
  - RNN emergence = learning a lookup TABLE (input-output mapping)

The quality of emergence is not just predictive accuracy—it is whether
the model instantiates an autonomous dynamical system that mirrors
the generative process of the world.
""")


if __name__ == "__main__":
    main()
