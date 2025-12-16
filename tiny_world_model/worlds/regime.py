# tiny_world_model/worlds/regime.py

import torch
from torch.utils.data import DataLoader, TensorDataset


class RegimeMarkovWorld:
    """
    E2 structured world:
      - Hidden regime_t ∈ {0,1}
      - Observed state_t ∈ {0,1}
      - Each regime has different P(same | regime)
    """
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.T = cfg.regime.T
        self.p_same0 = cfg.regime.p_same_regime0
        self.p_same1 = cfg.regime.p_same_regime1
        self.switch_type = cfg.regime.regime_switch.type
        self.period = cfg.regime.regime_switch.get("period", 32)

        self.n_train = cfg.data_gen.n_train_seqs
        self.n_eval  = cfg.data_gen.n_eval_seqs

        self.train_latent, self.train_obs = self._gen_dataset(self.n_train)
        self.eval_latent,  self.eval_obs  = self._gen_dataset(self.n_eval)

        # Inputs: [state, t/T] — no normalization needed
        self.train_input = self._make_input(self.train_obs)
        self.eval_input  = self._make_input(self.eval_obs)

        # baseline for NLL: global mean predictor
        self.baseline_mean = self.train_obs.mean().item()

    def _gen_dataset(self, n_seqs):
        device = self.device
        T = self.T

        regime = torch.zeros(n_seqs, T, 1, device=device)  # 0 or 1
        state  = torch.zeros(n_seqs, T, 1, device=device)  # 0 or 1

        # initial regime, state
        regime[:, 0] = torch.randint(0, 2, (n_seqs, 1), device=device).float()
        state[:, 0]  = torch.randint(0, 2, (n_seqs, 1), device=device).float()

        for t in range(1, T):
            # schedule regime switches
            if self.switch_type == "periodic" and t % self.period == 0:
                regime[:, t] = 1 - regime[:, t-1]
            else:
                regime[:, t] = regime[:, t-1]

            # pick P(same) based on regime
            p_same = torch.where(regime[:, t] == 0,
                                 torch.full_like(state[:, t-1], self.p_same0),
                                 torch.full_like(state[:, t-1], self.p_same1))
            same = torch.bernoulli(p_same).to(device)
            state[:, t] = torch.where(same == 1, state[:, t-1], 1 - state[:, t-1])

        return regime, state

    def _make_input(self, state):
        # state: [N, T, 1], convert to float + time_cond
        N, T, _ = state.shape
        device = state.device
        s = state.float()
        t = torch.arange(T, device=device).float() / (T - 1)
        t = t.view(1, T, 1).expand(N, T, 1)
        return torch.cat([s, t], dim=-1)  # [N, T, 2]

    def train_dataloader(self, batch_size):
        ds = TensorDataset(self.train_input, self.train_obs, self.train_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def eval_dataloader(self, batch_size):
        ds = TensorDataset(self.eval_input, self.eval_obs, self.eval_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


class RegimeNullWorld:
    """
    E2 null world:
      - No regime; i.i.d. Bernoulli state_t with given marginal p.
      - Latent regime is meaningless (all zeros).
    """
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.T = cfg.regime.T
        self.p_one = cfg.regime.marginal_p_one
        self.n_train = cfg.data_gen.n_train_seqs
        self.n_eval  = cfg.data_gen.n_eval_seqs

        self.train_latent, self.train_obs = self._gen_dataset(self.n_train)
        self.eval_latent,  self.eval_obs  = self._gen_dataset(self.n_eval)

        self.train_input = self._make_input(self.train_obs)
        self.eval_input  = self._make_input(self.eval_obs)

        # baseline for NLL
        self.baseline_mean = self.train_obs.mean().item()

    def _gen_dataset(self, n_seqs):
        device = self.device
        T = self.T
        probs = torch.full((n_seqs, T, 1), self.p_one, device=device)
        state = torch.bernoulli(probs)
        latent = torch.zeros_like(state)   # no regimes
        return latent, state

    def _make_input(self, state):
        N, T, _ = state.shape
        device = state.device
        s = state.float()
        t = torch.arange(T, device=device).float() / (T - 1)
        t = t.view(1, T, 1).expand(N, T, 1)
        return torch.cat([s, t], dim=-1)  # [N, T, 2]

    def train_dataloader(self, batch_size):
        ds = TensorDataset(self.train_input, self.train_obs, self.train_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def eval_dataloader(self, batch_size):
        ds = TensorDataset(self.eval_input, self.eval_obs, self.eval_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
