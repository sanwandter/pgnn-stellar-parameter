import torch
import torch.nn as nn

from pgnn.config import LOGG_GRID


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if downsample or stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class ResNet1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)

        self.resblock1 = ResBlock1D(32, 32, stride=1, downsample=False)
        self.resblock2 = ResBlock1D(32, 64, stride=1, downsample=True)
        self.resblock3 = ResBlock1D(64, 128, stride=1, downsample=True)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.relu_fc = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.gap(x).squeeze(-1)
        return self.relu_fc(self.bn_fc(self.fc(x)))


def _make_head(latent=64, hidden=128, dropout=0.3):
    return nn.Sequential(
        nn.Linear(latent, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


class MultiTaskHeads(nn.Module):
    def __init__(self, latent=64, dropout=0.3):
        super().__init__()
        self.head_teff = _make_head(latent, 128, dropout)
        self.head_logg = _make_head(latent, 128, dropout)
        self.head_logmdot = _make_head(latent, 128, dropout)
        self.head_rstar = _make_head(latent, 128, dropout)

    def forward(self, z):
        return {
            "teff": self.head_teff(z),
            "logg": self.head_logg(z),
            "logmdot": self.head_logmdot(z),
            "rstar": self.head_rstar(z),
        }


class PGNNModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = ResNet1DEncoder()
        self.heads = MultiTaskHeads(latent=64, dropout=dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.heads(z)


def snap_logg(logg_phys: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(LOGG_GRID, dtype=logg_phys.dtype, device=logg_phys.device)
    diffs = torch.abs(grid.unsqueeze(0) - logg_phys.view(-1, 1))
    idx = diffs.argmin(dim=1)
    return grid[idx]
