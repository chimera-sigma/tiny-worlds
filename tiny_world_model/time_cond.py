# tiny_world_model/time_cond.py
import torch

def apply_time_cond(x, time_cfg, T: int):
    """
    x: [B, T, d_x]  (already normalized obs features)
    returns: [B, T, d_x + d_time]
    """
    B, T_, d = x.shape
    assert T_ == T
    device = x.device

    t = torch.arange(T, device=device).float() / (T - 1)
    t = t.view(1, T, 1).expand(B, T, 1)

    if time_cfg.type == "full_time":
        t_feat = t
    elif time_cfg.type == "no_time":
        t_feat = torch.zeros_like(t)
    elif time_cfg.type == "constant_time":
        t_feat = torch.full_like(t, time_cfg.get("value", 0.5))
    elif time_cfg.type == "random_time":
        # Important: random mapping PER BATCH, PER CALL
        perm = torch.stack([torch.randperm(T, device=device) for _ in range(B)], dim=0)
        t_feat = t.gather(1, perm.unsqueeze(-1))
    else:
        raise ValueError(f"Unknown time_cond type: {time_cfg.type}")

    return torch.cat([x, t_feat], dim=-1)
