#!/usr/bin/env python3
"""
Lyapunov Spectrum Analysis for RNN/GRU hidden dynamics.

Key question: Are the learned dynamics "selective contraction" (Hamiltonian-like
flow with λ1 ≈ 0) or "pure dissipation" (all λ < 0)?

For a conformal symplectic system on a limit cycle:
- λ1 ≈ 0 (neutral stability along the flow/phase direction)
- λ2..N < 0 (contraction onto the manifold)

For a pure dissipative sink:
- All λ < 0 (everything contracts, including the flow direction)

Usage:
    python scripts/analyze_lyapunov_spectrum.py world=E3_osc_struct model.type=gru
    python scripts/analyze_lyapunov_spectrum.py world=E3_osc_struct model.type=rnn_tanh
"""

import torch
import torch.nn.functional as F
import numpy as np

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
    """Factory function for creating models."""
    if model_type == "rnn_tanh":
        return TinyRNN(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=init_cfg).to(device)
    elif model_type == "gru":
        return TinyGRU(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=init_cfg).to(device)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")


def compute_lyapunov_spectrum(model, x_seq, cfg, device, T_warmup=50, n_steps=200):
    """
    Compute Lyapunov exponents using QR decomposition (Benettin's algorithm).

    Args:
        model: Trained RNN/GRU
        x_seq: Input sequence [1, T, d_in]
        cfg: Config for time conditioning
        device: torch device
        T_warmup: Steps to settle onto attractor
        n_steps: Steps to accumulate spectrum

    Returns:
        lyap_exponents: Array of H exponents (sorted largest to smallest)
    """
    model.train()  # Need gradients through RNN

    # Get hidden dim
    H = model.hidden_dim

    # Initialize hidden state
    h = torch.zeros(1, H, device=device)

    # Q matrix tracks perturbation orientation
    Q = torch.eye(H, device=device)

    log_r_sums = np.zeros(H)
    valid_steps = 0

    T_total = min(T_warmup + n_steps, x_seq.shape[1] - 1)

    # Warmup phase - just run forward to settle on attractor
    with torch.no_grad():
        for t in range(T_warmup):
            x_t = x_seq[:, t, :]  # [1, d_in]
            h = model.step(h, x_t)

    # Measurement phase
    for t in range(T_warmup, T_total):
        x_t = x_seq[:, t, :].detach()  # [1, d_in]

        # Compute Jacobian J = dh_{t+1}/dh_t
        h_in = h.clone().detach().requires_grad_(True)
        h_next = model.step(h_in, x_t)

        # Build Jacobian column by column
        jac_cols = []
        for i in range(H):
            grad_outputs = torch.zeros_like(h_next)
            grad_outputs[0, i] = 1.0
            grad = torch.autograd.grad(h_next, h_in, grad_outputs=grad_outputs,
                                       retain_graph=True, create_graph=False)[0]
            jac_cols.append(grad.view(-1))

        J = torch.stack(jac_cols, dim=0)  # [H, H] - rows are output dims

        # Update perturbation matrix: M = J @ Q
        M = torch.matmul(J, Q)

        # QR decomposition
        Q_new, R = torch.linalg.qr(M)

        # Accumulate log of diagonal elements (expansion rates)
        diag_R = torch.diag(R)
        # Handle sign: take absolute value for logs
        log_diag = torch.log(torch.abs(diag_R) + 1e-12)
        log_r_sums += log_diag.detach().cpu().numpy()

        # Update for next iteration
        h = h_next.detach()
        Q = Q_new.detach()
        valid_steps += 1

    if valid_steps == 0:
        return np.zeros(H)

    # Average and sort (largest first)
    lyap_exponents = log_r_sums / valid_steps
    lyap_exponents = np.sort(lyap_exponents)[::-1]  # Descending order

    return lyap_exponents


def interpret_spectrum(spectrum, model_name, tolerance=0.1):
    """Interpret the Lyapunov spectrum."""
    l1 = spectrum[0]
    l_sum = np.sum(spectrum)

    print(f"\n{'='*60}")
    print(f"LYAPUNOV SPECTRUM: {model_name}")
    print(f"{'='*60}")
    print(f"Full spectrum: {np.array2string(spectrum, precision=3, separator=', ')}")
    print(f"Leading exponent (λ1): {l1:.4f}")
    print(f"Sum (≈ log|det J|):    {l_sum:.4f}")
    print(f"Remaining (λ2..N):     {spectrum[1:].mean():.4f} (mean)")

    print(f"\n--- INTERPRETATION ---")
    if abs(l1) < tolerance:
        print(f"λ1 ≈ 0 (within ±{tolerance})")
        print(">> MARGINALLY STABLE along flow direction")
        print(">> Suggests HAMILTONIAN-LIKE dynamics preserved on manifold")
        print(">> Selective contraction: geometry preserved, noise killed")
    elif l1 < -tolerance:
        print(f"λ1 < -{tolerance}")
        print(">> STABLE SINK - all directions contracting")
        print(">> Pure dissipation without preserved flow structure")
        print(">> Model is 'crushing' the state space")
    else:
        print(f"λ1 > {tolerance}")
        print(">> CHAOTIC - expansion present")
        print(">> Unstable dynamics")

    # Check if there's a gap between λ1 and λ2
    if len(spectrum) > 1:
        gap = spectrum[0] - spectrum[1]
        print(f"\nSpectral gap (λ1 - λ2): {gap:.4f}")
        if gap > 0.5 and abs(l1) < tolerance:
            print(">> Large gap with neutral λ1 suggests clean separation")
            print(">> Flow direction distinct from contraction directions")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("="*60)
    print("LYAPUNOV SPECTRUM ANALYSIS")
    print("="*60)
    print("\nConfig:")
    print(f"  World: {cfg.world.id}")
    print(f"  Model: {cfg.model.type}")
    print(f"  Hidden dim: {cfg.model.hidden_dim}")

    device = torch.device("cpu")  # CPU for stability with autograd
    set_all_seeds(cfg.random.master_seed)

    # Build world
    world = OscillatorWorld(cfg.world, device=device)

    # Model setup
    input_dim = cfg.world.input_dim
    hidden_dim = cfg.model.hidden_dim
    model_type = cfg.model.type

    # Create and train model
    model = create_model(model_type, input_dim, hidden_dim, cfg.model.init, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.optimizer.lr)

    print(f"\n[Training {model_type} on structured oscillator...]")
    train_loader = world.train_dataloader(cfg.train.batch_size)
    max_steps = 5000
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
                break
        if step >= max_steps:
            break

    print(f"\n[Computing Lyapunov spectrum...]")

    # Get a long evaluation sequence
    eval_loader = world.eval_dataloader(batch_size=1)
    x_in, y, z_lat = next(iter(eval_loader))
    x_in = x_in.to(device)
    x_tc = apply_time_cond(x_in, cfg.time_cond, world.T)

    # Compute spectrum for trained model
    spectrum_trained = compute_lyapunov_spectrum(
        model, x_tc, cfg, device,
        T_warmup=50, n_steps=150
    )
    interpret_spectrum(spectrum_trained, f"TRAINED {model_type.upper()}")

    # Also compute for random model (baseline)
    print(f"\n[Computing spectrum for RANDOM {model_type}...]")
    model_random = create_model(model_type, input_dim, hidden_dim, cfg.model.init, device)
    spectrum_random = compute_lyapunov_spectrum(
        model_random, x_tc, cfg, device,
        T_warmup=50, n_steps=150
    )
    interpret_spectrum(spectrum_random, f"RANDOM {model_type.upper()}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"                    Random        Trained")
    print(f"λ1 (flow dir):      {spectrum_random[0]:+.4f}       {spectrum_trained[0]:+.4f}")
    print(f"λ_sum (log|det|):   {spectrum_random.sum():+.4f}       {spectrum_trained.sum():+.4f}")
    print(f"λ2..N mean:         {spectrum_random[1:].mean():+.4f}       {spectrum_trained[1:].mean():+.4f}")

    if abs(spectrum_trained[0]) < 0.1 and spectrum_trained[1:].mean() < -0.3:
        print("\n>> VERDICT: Selective contraction (Hamiltonian-like)")
        print("   Flow direction preserved, transverse directions contracted")
    elif spectrum_trained[0] < -0.1:
        print("\n>> VERDICT: Pure dissipation (stable sink)")
        print("   All directions contracted, including flow")
    else:
        print("\n>> VERDICT: Unclear pattern")


if __name__ == "__main__":
    main()
