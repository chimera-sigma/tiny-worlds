# tiny_world_model/worlds/oscillator.py

import torch
from torch.utils.data import DataLoader, TensorDataset

class OscillatorWorld:
    """
    E3 structured oscillator world:
      - Latent: 2D nonlinear oscillator (z_t = (x_t, y_t))
      - Observation: x_t + Gaussian noise
      - RNN input: normalized obs + time_cond (added in run_experiment via apply_time_cond)
    """
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.dt        = cfg.oscillator.dt
        self.T         = cfg.oscillator.T
        self.sigma_obs = cfg.oscillator.sigma_obs
        self.omega     = cfg.oscillator.get("omega", 1.0)

        self.n_train = cfg.data_gen.n_train_seqs
        self.n_eval  = cfg.data_gen.n_eval_seqs

        # generate latent + obs
        self.train_latent, self.train_obs = self._gen_dataset(self.n_train)
        self.eval_latent,  self.eval_obs  = self._gen_dataset(self.n_eval)

        # global z-score normalization on training obs only
        mu = self.train_obs.mean()
        std = self.train_obs.std() + 1e-6
        self.mu, self.std = mu, std

        self.train_input = (self.train_obs - mu) / std
        self.eval_input  = (self.eval_obs  - mu) / std

        # baseline mean for smart trivial baseline
        self.baseline_mean = self.train_obs.mean().item()

    @property
    def T_steps(self):
        return self.T

    def _hopf_step(self, z):
        """
        Simple Hopf-like oscillator step in R^2:
          z = (x, y)
          dx/dt = (1 - r^2) * x - omega * y
          dy/dt = (1 - r^2) * y + omega * x
        This produces a stable limit cycle.
        """
        x, y = z[..., 0], z[..., 1]
        r2 = x * x + y * y
        dx = (1.0 - r2) * x - self.omega * y
        dy = (1.0 - r2) * y + self.omega * x

        x_next = x + self.dt * dx
        y_next = y + self.dt * dy
        return torch.stack([x_next, y_next], dim=-1)

    def _gen_dataset(self, n_seqs):
        """
        Generate (latent, obs) pairs:
          latent: [N, T, 2]
          obs:    [N, T, 1], where obs_t = x_t + noise
        """
        device = self.device
        T = self.T

        # small random initial conditions near origin
        z = torch.randn(n_seqs, 2, device=device) * 0.1
        latents = torch.zeros(n_seqs, T, 2, device=device)
        obs     = torch.zeros(n_seqs, T, 1, device=device)

        for t in range(T):
            latents[:, t] = z
            x_t = z[:, 0]
            obs[:, t, 0] = x_t + torch.randn(n_seqs, device=device) * self.sigma_obs
            z = self._hopf_step(z)

        return latents, obs

    def train_dataloader(self, batch_size):
        """
        Returns (x_in, y, z_latent):
          x_in = normalized obs  [B, T, 1]
          y    = raw obs         [B, T, 1]
          z    = latent          [B, T, 2]
        """
        ds = TensorDataset(self.train_input, self.train_obs, self.train_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def eval_dataloader(self, batch_size):
        ds = TensorDataset(self.eval_input, self.eval_obs, self.eval_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


class OscillatorNullWorld(OscillatorWorld):
    """
    Null version of OscillatorWorld:
    - same oscillator parameters, but observations are temporally shuffled
      so there is no usable sequential structure or phase coherence.
    """
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device=device)
        # overwrite train/eval obs with shuffled versions
        self.train_obs = self._shuffle_time(self.train_obs)
        self.eval_obs  = self._shuffle_time(self.eval_obs)
        # latent no longer meaningful; just zeros
        self.train_latent = torch.zeros_like(self.train_latent)
        self.eval_latent  = torch.zeros_like(self.eval_latent)

        # recompute normalized inputs based on shuffled obs
        # (mu, std already computed from original train_obs in parent __init__)
        self.train_input = (self.train_obs - self.mu) / self.std
        self.eval_input  = (self.eval_obs  - self.mu) / self.std

        # baseline_mean stays the same (global mean unchanged by shuffling)

    def _shuffle_time(self, y):
        """
        y: [N, T, 1] -> same shape, but time shuffled per sequence.
        """
        N, T, D = y.shape
        device = y.device
        y_shuf = y.clone()
        for i in range(N):
            perm = torch.randperm(T, device=device)
            y_shuf[i] = y[i, perm]
        return y_shuf


class OscillatorNonstatWorld(OscillatorWorld):
    """
    Non-stationary oscillator:
      - omega changes linearly from omega_start to omega_end over time.
      - Tests whether operator emergence is robust to slow drift in dynamics.
    """
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.dt        = cfg.oscillator.dt
        self.T         = cfg.oscillator.T
        self.sigma_obs = cfg.oscillator.sigma_obs
        self.omega_start = cfg.oscillator.omega_start
        self.omega_end   = cfg.oscillator.omega_end

        self.n_train = cfg.data_gen.n_train_seqs
        self.n_eval  = cfg.data_gen.n_eval_seqs

        self.train_latent, self.train_obs = self._gen_dataset(self.n_train)
        self.eval_latent,  self.eval_obs  = self._gen_dataset(self.n_eval)

        # global z-score
        mu = self.train_obs.mean()
        std = self.train_obs.std() + 1e-6
        self.mu, self.std = mu, std

        self.train_input = (self.train_obs - mu) / std
        self.eval_input  = (self.eval_obs  - mu) / std

        self.baseline_mean = self.train_obs.mean().item()

    @property
    def T_steps(self):
        return self.T

    def _gen_dataset(self, n_seqs):
        device = self.device
        T = self.T

        z = torch.randn(n_seqs, 2, device=device) * 0.1
        latents = torch.zeros(n_seqs, T, 2, device=device)
        obs     = torch.zeros(n_seqs, T, 1, device=device)

        for t in range(T):
            # interpolate omega over time
            alpha = t / (T - 1)
            omega_t = self.omega_start + alpha * (self.omega_end - self.omega_start)

            latents[:, t] = z
            x_t = z[:, 0]
            obs[:, t, 0] = x_t + torch.randn(n_seqs, device=device) * self.sigma_obs

            # step with time-varying omega
            z = self._hopf_step_timevarying(z, omega_t)

        return latents, obs

    def _hopf_step_timevarying(self, z, omega_t):
        """
        Hopf step with time-varying omega.
        """
        x, y = z[..., 0], z[..., 1]
        r2 = x * x + y * y
        dx = (1.0 - r2) * x - omega_t * y
        dy = (1.0 - r2) * y + omega_t * x
        x_next = x + self.dt * dx
        y_next = y + self.dt * dy
        return torch.stack([x_next, y_next], dim=-1)

    def train_dataloader(self, batch_size):
        ds = TensorDataset(self.train_input, self.train_obs, self.train_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def eval_dataloader(self, batch_size):
        ds = TensorDataset(self.eval_input, self.eval_obs, self.eval_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
