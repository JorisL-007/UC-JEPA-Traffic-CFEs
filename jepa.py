import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, adj, in_dim, out_dim):
        super().__init__()
        a = adj + torch.eye(adj.shape[0], device=adj.device)
        d_inv_sqrt = a.sum(dim=1).pow(-0.5)
        a_hat = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
        self.register_buffer("a_hat", a_hat)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):  # x: (B, T, N, F)
        x = torch.einsum("nm,btmf->btnf", self.a_hat, x)
        return F.relu(self.lin(x))


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, adj, in_dim, d=64):
        super().__init__()
        self.gcn = GCNLayer(adj, in_dim, d)
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, x):
        B, T, N, _ = x.shape
        h = self.gcn(x)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, last = self.gru(h)
        return last.squeeze(0).view(B, N, -1)


class Predictor(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, z):
        return self.net(z)


class TrafficJEPA(nn.Module):
    def __init__(self, adj, in_dim=20, d=64, out_steps=12):
        super().__init__()
        self.context_encoder = SpatioTemporalEncoder(adj, in_dim, d)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(d)
        self.speed_head = nn.Linear(d, out_steps)

    def latent_loss(self, x_ctx, x_fut):
        z_ctx = self.context_encoder(x_ctx)
        with torch.no_grad():
            z_tgt = self.target_encoder(x_fut)
        z_pred = self.predictor(z_ctx)
        return F.mse_loss(z_pred, z_tgt)

    def forecast(self, x_ctx):
        z_pred = self.predictor(self.context_encoder(x_ctx))
        return self.speed_head(z_pred).transpose(1, 2)

    @torch.no_grad()
    def ema_update(self, momentum=0.996):
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(momentum).add_(pc, alpha=1 - momentum)


if __name__ == "__main__":
    N = 207
    adj = (torch.rand(N, N) < 0.05).float()
    model = TrafficJEPA(adj)

    x_ctx = torch.randn(4, 12, N, 20)
    x_fut = torch.randn(4, 12, N, 20)

    print("latent loss:", model.latent_loss(x_ctx, x_fut).item())
    print("forecast shape:", tuple(model.forecast(x_ctx).shape))