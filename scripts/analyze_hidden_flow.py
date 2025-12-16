# scripts/analyze_hidden_flow.py

import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_world_model.worlds.oscillator import OscillatorWorld, OscillatorNullWorld
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


def compute_jacobian_stats(model, world, cfg, device, n_samples=32, t_index=100):
    """
    For a given model + world:
      - sample a batch from eval,
      - choose a timestep t_index,
      - compute J = d h_{t+1} / d h_t for n_samples,
      - return mean and std of log |det J|.
    """
    # For Jacobian computation, we need the RNN in "training" mode
    # so that cudnn allows backward passes.
    model.train()
    model.to(device)

    eval_loader = world.eval_dataloader(batch_size=n_samples)
    x_in, y, z_lat = next(iter(eval_loader))
    x_in = x_in.to(device)
    # add time conditioning
    x_tc = apply_time_cond(x_in, cfg.time_cond, world.T)

    # forward to get full h_seq
    with torch.no_grad():
        _, h_seq = model(x_tc)     # h_seq: [B, T, H]

    B, T, H = h_seq.shape
    t = min(max(1, t_index), T - 2)   # safety clamp
    h_t = h_seq[:, t, :]              # [B, H]
    x_t = x_tc[:, t, :]               # [B, d_in]

    # For each sample i, compute J_i: [H, H]
    log_abs_dets = []

    for i in range(min(n_samples, B)):
        h_i = h_t[i:i+1].clone().detach().requires_grad_(True)   # [1, H]
        x_i = x_t[i:i+1].clone().detach()                        # [1, d_in]

        def F_hi(h_flat):
            # h_flat: [H], we need [1, H]
            h = h_flat.view(1, H)
            h_next = model.step(h, x_i)           # [1, H]
            return h_next.view(-1)                # [H]

        # Use autograd.functional.jacobian on flattened vector
        with torch.enable_grad():
            J = jacobian(F_hi, h_i.view(-1))      # [H, H]
        J = J.detach().cpu()

        # Compute log |det J|
        try:
            det = torch.det(J)
            log_abs_det = torch.log(torch.abs(det) + 1e-12).item()
        except RuntimeError:
            # singular or numerical issue → treat as large negative
            log_abs_det = float('-inf')

        log_abs_dets.append(log_abs_det)

    log_abs_dets = torch.tensor(log_abs_dets)
    mean_val = log_abs_dets[torch.isfinite(log_abs_dets)].mean().item()
    std_val  = log_abs_dets[torch.isfinite(log_abs_dets)].std().item()
    return mean_val, std_val


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    set_all_seeds(cfg.random.master_seed)

    # --- build worlds ---
    # We bypass Hydra group for world here and instantiate directly
    osc_struct_cfg = cfg.world
    osc_struct_cfg.id = "E3_osc_struct"
    osc_struct_cfg.type = "oscillator"

    world_struct = OscillatorWorld(osc_struct_cfg, device=device)
    world_null   = OscillatorNullWorld(osc_struct_cfg, device=device)  # uses same cfg, but shuffles

    # --- create models ---
    input_dim  = osc_struct_cfg.input_dim
    hidden_dim = cfg.model.hidden_dim
    model_type = cfg.model.type  # "rnn_tanh" or "gru"

    print(f"\n[Model type: {model_type}, hidden_dim: {hidden_dim}]")

    # 1) random reservoir
    model_random = create_model(model_type, input_dim, hidden_dim, cfg.model.init, device)

    # 2) trained on structured oscillator
    model_struct = create_model(model_type, input_dim, hidden_dim, cfg.model.init, device)
    opt = torch.optim.Adam(model_struct.parameters(), lr=cfg.model.optimizer.lr)

    train_loader = world_struct.train_dataloader(cfg.train.batch_size)
    max_steps = 5000  # small training for this analysis
    step = 0
    print("\n[Training] Structured oscillator model...")
    for epoch in range(9999):
        for x_in, y, z_lat in train_loader:
            x_in = x_in.to(device)
            y    = y.to(device)
            x_tc = apply_time_cond(x_in, cfg.time_cond, world_struct.T)

            pred, h_seq = model_struct(x_tc)
            loss = F.mse_loss(pred[:, :-1], y[:, 1:])

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step % 1000 == 0:
                print(f"  step {step}, loss={loss.item():.4f}")
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    # 3) trained on null oscillator
    model_null = create_model(model_type, input_dim, hidden_dim, cfg.model.init, device)
    opt_null = torch.optim.Adam(model_null.parameters(), lr=cfg.model.optimizer.lr)

    train_loader_null = world_null.train_dataloader(cfg.train.batch_size)
    step = 0
    print("\n[Training] Null oscillator model...")
    for epoch in range(9999):
        for x_in, y, z_lat in train_loader_null:
            x_in = x_in.to(device)
            y    = y.to(device)
            x_tc = apply_time_cond(x_in, cfg.time_cond, world_null.T)

            pred, h_seq = model_null(x_tc)
            loss = F.mse_loss(pred[:, :-1], y[:, 1:])

            opt_null.zero_grad()
            loss.backward()
            opt_null.step()

            step += 1
            if step % 1000 == 0:
                print(f"  step {step}, loss={loss.item():.4f}")
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    # --- analyze Jacobian stats ---
    print("\n" + "="*60)
    print(f"JACOBIAN ANALYSIS: log |det J| statistics ({model_type.upper()}, H={hidden_dim})")
    print("="*60)

    print(f"\n[1] Random {model_type} (untrained, on E3_struct data):")
    m, s = compute_jacobian_stats(model_random, world_struct, cfg, device)
    print(f"  mean log |det J| = {m:.4f}, std = {s:.4f}")

    print(f"\n[2] {model_type} trained on structured oscillator:")
    m, s = compute_jacobian_stats(model_struct, world_struct, cfg, device)
    print(f"  mean log |det J| = {m:.4f}, std = {s:.4f}")

    print(f"\n[3] {model_type} trained on null oscillator (shuffled time):")
    m, s = compute_jacobian_stats(model_null, world_struct, cfg, device)
    print(f"  mean log |det J| = {m:.4f}, std = {s:.4f}")

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("  - Closer to 0 = more volume-preserving (symplectic-like)")
    print("  - More negative = stronger contraction (attractor dynamics)")
    print("  - Lower std = more regular/stable flow")
    print("="*60)


if __name__ == "__main__":
    main()
