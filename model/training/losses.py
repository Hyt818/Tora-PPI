'''
Purpose: Defines the alignment and regularization loss functions used in the Tora-PPI training, including InfoNCE, Barlow Twins, and VICReg.
Input: Intermediate representation tensors of sequence-view, structure-view, or graph-view.
Output: Scalar loss, used in conjunction with the PPI multi-label classification loss to optimize the model.
'''

import torch
import torch.nn.functional as F

def info_nce_loss(z_gnn, z_seq, tau=0.07):
    logits = torch.matmul(z_gnn, z_seq.t()) / tau
    labels = torch.arange(z_gnn.size(0), device=z_gnn.device)

    loss_g2s = F.cross_entropy(logits, labels)
    loss_s2g = F.cross_entropy(logits.t(), labels)

    return (loss_g2s + loss_s2g) / 2


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def covariance(z: torch.Tensor) -> torch.Tensor:
    n = z.size(0)
    return (z.T @ z) / (n - 1)


def barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambd: float = 5e-3,
    eps: float = 1e-9
) -> torch.Tensor:
    assert z1.shape == z2.shape
    N, D = z1.shape

    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + eps)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + eps)

    c = (z1.T @ z2) / N

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()

    return on_diag + lambd * off_diag


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4
) -> torch.Tensor:
    assert z1.shape == z2.shape
    N, D = z1.shape

    sim_loss = F.mse_loss(z1, z2)

    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)

    std_z1 = torch.sqrt(z1_centered.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2_centered.var(dim=0) + eps)

    var_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    cov_z1 = (z1_centered.T @ z1_centered) / (N - 1)
    cov_z2 = (z2_centered.T @ z2_centered) / (N - 1)

    cov_loss = off_diagonal(cov_z1).pow_(2).sum() / D
    cov_loss = cov_loss + off_diagonal(cov_z2).pow_(2).sum() / D

    return sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss
