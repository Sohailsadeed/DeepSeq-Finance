"""PyTorch sequence model architectures for financial time-series forecasting.
Implements SimpleRNN, LSTM, GRU, Attention, TCN, and TimeGAN networks."""

import torch
from torch import nn

import config


class _RecurrentBase(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 1,
                 hidden_size=config.HIDDEN_SIZE,
                 num_layers=config.NUM_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim),
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = n_features
        self.dropout = dropout
        self.out_dim = out_dim

    def _head(self, last):
        out = self.head(last)
        return out.squeeze(-1) if self.out_dim == 1 else out


class SimpleRNNNet(_RecurrentBase):
    def __init__(self, n_features: int, out_dim: int = 1, **kw):
        super().__init__(n_features, out_dim=out_dim, **kw)
        self.rnn = nn.RNN(n_features, self.hidden_size, self.num_layers,
                          batch_first=True, dropout=self.dropout if self.num_layers > 1 else 0.0)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self._head(out[:, -1, :])


class LSTMNet(_RecurrentBase):
    def __init__(self, n_features: int, out_dim: int = 1, **kw):
        super().__init__(n_features, out_dim=out_dim, **kw)
        self.rnn = nn.LSTM(n_features, self.hidden_size, self.num_layers,
                           batch_first=True, dropout=self.dropout if self.num_layers > 1 else 0.0)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self._head(out[:, -1, :])


class GRUNet(_RecurrentBase):
    def __init__(self, n_features: int, out_dim: int = 1, **kw):
        super().__init__(n_features, out_dim=out_dim, **kw)
        self.rnn = nn.GRU(n_features, self.hidden_size, self.num_layers,
                          batch_first=True, dropout=self.dropout if self.num_layers > 1 else 0.0)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self._head(out[:, -1, :])


class LSTMAttentionNet(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 1,
                 hidden_size=config.HIDDEN_SIZE,
                 num_layers=config.NUM_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = n_features
        self.dropout = dropout
        self.out_dim = out_dim

        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn_energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )
        self.last_attn_weights = None

    def forward(self, x, return_attn: bool = False):
        out, _ = self.lstm(x)
        scores = self.attn_energy(out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        self.last_attn_weights = weights.detach()
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        logits = self.head(context)
        res = logits.squeeze(-1) if self.out_dim == 1 else logits
        if return_attn:
            return res, weights
        return res


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=config.DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNet(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 1,
                 hidden_size=config.HIDDEN_SIZE,
                 num_layers=2, kernel_size=3, dropout=config.DROPOUT):
        super().__init__()
        self.out_dim = out_dim
        layers = []
        num_channels = [hidden_size] * num_layers
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = n_features if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x):
        x_trans = x.transpose(1, 2)
        y = self.network(x_trans)
        last = y[:, :, -1]
        out = self.head(last)
        return out.squeeze(-1) if self.out_dim == 1 else out


class TimeGANGenerator(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 1,
                 hidden_size=config.HIDDEN_SIZE,
                 num_layers=config.NUM_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.rnn = nn.GRU(n_features, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, out_dim)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        res = self.fc(last)
        return res.squeeze(-1) if self.out_dim == 1 else res


class TimeGANDiscriminator(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 1,
                 hidden_size=config.HIDDEN_SIZE, dropout=config.DROPOUT):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + out_dim, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        self.rnn = nn.GRU(n_features, hidden_size, 1, batch_first=True)

    def forward(self, x, y):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        inp = torch.cat([last, y], dim=-1)
        return self.fc(inp).squeeze(-1)


class TimeGANNet(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 1, **kw):
        super().__init__()
        self.generator = TimeGANGenerator(n_features, out_dim=out_dim, **kw)
        self.discriminator = TimeGANDiscriminator(n_features, out_dim=out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        return self.generator(x)


MODEL_REGISTRY = {
    "SimpleRNN": SimpleRNNNet,
    "LSTM": LSTMNet,
    "GRU": GRUNet,
    "LSTM_Attention": LSTMAttentionNet,
    "TCN": TCNNet,
    "TimeGAN": TimeGANNet,
}
