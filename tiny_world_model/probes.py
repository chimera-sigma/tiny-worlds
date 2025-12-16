# tiny_world_model/probes.py
import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def train_probe(H, Z, n_epochs=200, lr=1e-2, weight_decay=1e-3, patience=20):
    """
    H: [N, d_h], Z: [N, 1] (regression)
    """
    device = H.device
    if Z.ndim == 1:
        Z = Z.unsqueeze(-1)

    N, d_h = H.shape
    d_lat = Z.shape[1]

    idx = torch.randperm(N, device=device)
    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]
    H_train, Z_train = H[train_idx], Z[train_idx]
    H_val,   Z_val   = H[val_idx],   Z[val_idx]

    probe = LinearProbe(d_h, d_lat).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        probe.train()
        opt.zero_grad()
        pred = probe(H_train)
        loss = loss_fn(pred, Z_train)
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            pred_val = probe(H_val)
            val_loss = loss_fn(pred_val, Z_val).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    metrics = eval_probe_metrics(probe, H, Z)
    return probe, metrics


def eval_probe_metrics(probe, H, Z):
    probe.eval()
    with torch.no_grad():
        pred = probe(H)
    if Z.ndim == 1:
        Z = Z.unsqueeze(-1)
    mse = torch.mean((Z - pred) ** 2).item()
    var = torch.var(Z).item()

    if var < 1e-8:
        # R² undefined; target is almost constant
        r2 = float('nan')
    else:
        r2 = 1 - mse / (var + 1e-8)

    return {"r2": r2, "mse": mse}


def probe_random_hidden(H_shape, Z, device, **probe_kwargs):
    """
    Probe random hidden states (untrained model baseline).
    H_shape: (N, d_h) to generate random states
    Z: [N, d_lat] latent targets
    Returns: metrics dict with r2, mse
    """
    H_random = torch.randn(H_shape, device=device)
    _, metrics = train_probe(H_random, Z, **probe_kwargs)
    return metrics


def probe_permuted_hidden(H, Z, **probe_kwargs):
    """
    Probe permuted hidden states (shuffled across samples).
    H: [N, d_h] hidden states
    Z: [N, d_lat] latent targets
    Returns: metrics dict with r2, mse
    """
    idx_perm = torch.randperm(H.shape[0], device=H.device)
    H_perm = H[idx_perm]
    _, metrics = train_probe(H_perm, Z, **probe_kwargs)
    return metrics


def probe_with_controls(H_trained, Z, model_untrained=None, **probe_kwargs):
    """
    Run probe on trained hidden states + control baselines.

    Args:
        H_trained: [N, d_h] hidden states from trained model
        Z: [N, d_lat] latent targets
        model_untrained: optional untrained model to extract random hidden states
        **probe_kwargs: passed to train_probe

    Returns:
        dict with keys: trained, random, permuted (each with r2, mse)
    """
    # Main trained probe
    _, metrics_trained = train_probe(H_trained, Z, **probe_kwargs)

    # Random hidden states baseline
    metrics_random = probe_random_hidden(H_trained.shape, Z, H_trained.device, **probe_kwargs)

    # Permuted hidden states baseline
    metrics_permuted = probe_permuted_hidden(H_trained, Z, **probe_kwargs)

    return {
        "trained": metrics_trained,
        "random": metrics_random,
        "permuted": metrics_permuted
    }
