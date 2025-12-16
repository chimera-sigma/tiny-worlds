# tiny_world_model/worlds/drift.py
import torch
from torch.utils.data import DataLoader, TensorDataset

class DriftWorld:
    """
    E1 structured drift: latent x_t with constant drift, obs y_t = x_t + noise.
    RNN input uses per-sequence relative coordinates (y_t - y_0).
    """
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.T = cfg.drift.T
        self.v = cfg.drift.v
        self.sigma_dyn = cfg.drift.sigma_dyn
        self.sigma_obs = cfg.drift.sigma_obs

        self.n_train = cfg.data_gen.n_train_seqs
        self.n_eval  = cfg.data_gen.n_eval_seqs

        self.train_latent, self.train_obs = self._gen_dataset(self.n_train)
        self.eval_latent,  self.eval_obs  = self._gen_dataset(self.n_eval)

        # Global mean baseline from training data
        self.baseline_mean = self.train_obs.mean().item()

        self.train_input = self._make_rnn_input(self.train_obs)
        self.eval_input  = self._make_rnn_input(self.eval_obs)

    def _gen_dataset(self, n_seqs):
        T     = self.T
        v     = self.v
        s_dyn = self.sigma_dyn
        s_obs = self.sigma_obs
        device = self.device

        x = torch.zeros(n_seqs, T, 1, device=device)
        eps_dyn = torch.randn(n_seqs, T-1, 1, device=device) * s_dyn
        for t in range(T-1):
            x[:, t+1] = x[:, t] + v + eps_dyn[:, t]

        eps_obs = torch.randn_like(x) * s_obs
        y = x + eps_obs
        return x, y

    def _make_rnn_input(self, y):
        # y: [N, T, 1]
        y0 = y[:, 0:1, :]
        y_rel = y - y0
        return y_rel

    def train_dataloader(self, batch_size):
        ds = TensorDataset(self.train_input, self.train_obs, self.train_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def eval_dataloader(self, batch_size):
        ds = TensorDataset(self.eval_input, self.eval_obs, self.eval_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


class DriftNullWorld(DriftWorld):
    """
    Null version of DriftWorld:
    - same drift parameters, but observations are temporally shuffled
      so there is no usable sequential structure.
    """
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device=device)
        # overwrite train/eval obs with shuffled versions
        self.train_obs = self._shuffle_time(self.train_obs)
        self.eval_obs  = self._shuffle_time(self.eval_obs)
        # latent no longer meaningful; just zeros
        self.train_latent = torch.zeros_like(self.train_obs)
        self.eval_latent  = torch.zeros_like(self.eval_obs)
        # recompute inputs based on shuffled obs
        self.train_input = self._make_rnn_input(self.train_obs)
        self.eval_input  = self._make_rnn_input(self.eval_obs)

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


def make_E1_null_from_struct(world: DriftWorld):
    """
    Create a drift-null world: same obs distribution, shuffled in time.
    """
    y = world.train_obs.clone()
    N, T, _ = y.shape
    device = y.device

    y_null = y.clone()
    for i in range(N):
        perm = torch.randperm(T, device=device)
        y_null[i] = y[i, perm]

    w_null = object.__new__(DriftWorld)
    w_null.cfg = world.cfg
    w_null.device = world.device
    w_null.T = world.T
    w_null.v = 0.0
    w_null.sigma_dyn = world.sigma_dyn
    w_null.sigma_obs = world.sigma_obs
    w_null.n_train = world.n_train
    w_null.n_eval  = world.n_eval

    w_null.train_obs = y_null
    w_null.eval_obs  = world.eval_obs  # or reshuffle eval too, similar way
    w_null.train_latent = torch.zeros_like(y_null)
    w_null.eval_latent  = torch.zeros_like(world.eval_latent)

    w_null.train_input = w_null._make_rnn_input(w_null.train_obs)
    w_null.eval_input  = w_null._make_rnn_input(w_null.eval_obs)
    return w_null


class DriftDecoySineWorld:
    """
    Decoy: sine wave + noise. No underlying random-walk drift.
    """
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.T = cfg.drift.T
        self.omega = cfg.decoy_sine.omega
        self.amp = cfg.decoy_sine.amp
        self.sigma_obs = cfg.decoy_sine.sigma_obs

        self.n_train = cfg.data_gen.n_train_seqs
        self.n_eval  = cfg.data_gen.n_eval_seqs

        self.train_latent, self.train_obs = self._gen_dataset(self.n_train)
        self.eval_latent,  self.eval_obs  = self._gen_dataset(self.n_eval)

        # Global mean baseline from training data
        self.baseline_mean = self.train_obs.mean().item()

        self.train_input = self._make_rnn_input(self.train_obs)
        self.eval_input  = self._make_rnn_input(self.eval_obs)

    def _gen_dataset(self, n_seqs):
        device = self.device
        T = self.T
        t = torch.arange(T, device=device).view(1, T, 1)
        phases = torch.rand(n_seqs, 1, 1, device=device) * 2 * torch.pi
        latent = self.amp * torch.sin(self.omega * t + phases)  # [N, T, 1]
        obs = latent + torch.randn_like(latent) * self.sigma_obs
        return latent, obs

    def _make_rnn_input(self, y):
        y0 = y[:, 0:1, :]
        y_rel = y - y0
        return y_rel

    def train_dataloader(self, batch_size):
        ds = TensorDataset(self.train_input, self.train_obs, self.train_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def eval_dataloader(self, batch_size):
        ds = TensorDataset(self.eval_input, self.eval_obs, self.eval_latent)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
