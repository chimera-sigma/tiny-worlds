# tiny_world_model/models/rnn.py
import torch
import torch.nn as nn

class TinyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_cfg=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(hidden_dim, 1)  # predict next y (mean only for now)

        if init_cfg is not None and init_cfg.type == "orthogonal":
            for name, p in self.rnn.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(p, gain=init_cfg.gain)

    def forward(self, x):
        # x: [B, T, d_in]
        h_seq, _ = self.rnn(x)
        out = self.head(h_seq)  # [B, T, 1]
        return out, h_seq

    def step(self, h_t, x_t):
        """
        One-step transition:
          h_t: [B, H]
          x_t: [B, d_in]
        returns h_{t+1}: [B, H]
        """
        # nn.RNN expects [B, T=1, d_in] and initial hidden [1, B, H]
        x_t_seq = x_t.unsqueeze(1)                    # [B, 1, d_in]
        h0 = h_t.unsqueeze(0)                         # [1, B, H]
        h_seq, h_next = self.rnn(x_t_seq, h0)        # h_seq: [B, 1, H], h_next: [1, B, H]
        return h_next.squeeze(0)                     # [B, H]


class TinyGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_cfg=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

        # optional: orthogonal init on recurrent weights
        if init_cfg is not None and init_cfg.type == "orthogonal":
            for name, p in self.gru.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(p, gain=init_cfg.gain)

    def forward(self, x):
        h_seq, _ = self.gru(x)
        out = self.head(h_seq)
        return out, h_seq

    def step(self, h_t, x_t):
        """
        One-step transition:
          h_t: [B, H]
          x_t: [B, d_in]
        returns h_{t+1}: [B, H]
        """
        x_t_seq = x_t.unsqueeze(1)      # [B, 1, d_in]
        h0 = h_t.unsqueeze(0)           # [1, B, H]
        h_seq, h_next = self.gru(x_t_seq, h0)
        return h_next.squeeze(0)


class TinyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_cfg=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

        if init_cfg is not None and init_cfg.type == "orthogonal":
            for name, p in self.lstm.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(p, gain=init_cfg.gain)

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        out = self.head(h_seq)
        return out, h_seq

    def step(self, h_t, x_t, c_t=None):
        """
        One-step transition for LSTM:
          h_t: [B, H]
          x_t: [B, d_in]
          c_t: [B, H] (optional cell state)
        returns (h_{t+1}, c_{t+1}): ([B, H], [B, H])
        """
        x_t_seq = x_t.unsqueeze(1)      # [B, 1, d_in]
        if c_t is None:
            h0 = h_t.unsqueeze(0)       # [1, B, H]
            c0 = torch.zeros_like(h0)
        else:
            h0 = h_t.unsqueeze(0)
            c0 = c_t.unsqueeze(0)
        h_seq, (h_next, c_next) = self.lstm(x_t_seq, (h0, c0))
        return h_next.squeeze(0), c_next.squeeze(0)
