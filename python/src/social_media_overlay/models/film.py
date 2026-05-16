import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMResidualBlock(nn.Module):
    """
    Residual FiLM block for feature-wise modulation.
    https://ivadomed.org/technical_features.html
    https://arxiv.org/pdf/1709.07871
    https://distill.pub/2018/feature-wise-transformations/

    z:    image-conditioned latent, shape [B, feat_dim]
    cond: conditioning vector from alpha, shape [B, cond_dim]

    Output:
        z + modulated_update
    """

    def __init__(self, feat_dim, cond_dim, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = feat_dim
        if feat_dim < 1:
            raise ValueError(f"feat_dim must be >= 1, got {feat_dim}")
        if cond_dim < 1:
            raise ValueError(f"cond_dim must be >= 1, got {cond_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        self.fc = nn.Linear(feat_dim, feat_dim)

        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2 * feat_dim),
        )

        # Start near identity modulation:
        # gamma ~ 1, beta ~ 0
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, z, cond):
        h = self.fc(z)

        gamma_beta = self.to_gamma_beta(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        # Gentle bounded modulation
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        h = gamma * h + beta
        h = F.leaky_relu(h)

        return z + h
