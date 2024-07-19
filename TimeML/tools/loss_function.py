import torch
import torch.nn as nn

def mahalanobis_loss(output, target, covariance_matrix):
    delta = output - target
    inv_cov_matrix = torch.inverse(covariance_matrix)
    mahalanobis_dist = torch.matmul(torch.matmul(delta.unsqueeze(1), inv_cov_matrix), delta.unsqueeze(-1)).squeeze()
    loss = torch.mean(mahalanobis_dist)
    return loss

def euclidean_loss(output, target):
    return torch.mean((output - target) ** 2)

def manhattan_loss(output, target):
    return torch.mean(torch.abs(output - target))

def cosine_similarity_loss(output, target):
    cos_sim = nn.functional.cosine_similarity(output, target, dim=-1)
    return torch.mean(1 - cos_sim)