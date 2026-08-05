"""Loss functions: BPR, regularization, neighbor-aggregate, and critique losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def get_bpr_loss(user_embed, pos_embed, neg_embed):

    pos_scores = torch.sum(torch.mul(user_embed, pos_embed), dim=1)
    neg_scores = torch.sum(torch.mul(user_embed, neg_embed), dim=1)

    loss = -torch.log(torch.sigmoid(pos_scores-neg_scores) + 10e-8)

    return torch.mean(loss)

def get_critique_loss_base(user_embed, neg_embed):

    user_embed_expand = user_embed.unsqueeze(1)

    neg_score = torch.sum(user_embed_expand * neg_embed, dim=2)
    critique_loss = -1 * torch.mean(torch.log(1 - torch.sigmoid(neg_score) - 10e-8))

    return critique_loss

def get_critique_neg_loss_masked(user_embed, neg_embed, mask):
    # Masked negative critique loss (pushes down neg items).
    # user_embed [B,d], neg_embed [B,k,d], mask [B] (1 = user has neg critique this round).
    neg_score = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=2)        # [B,k]
    per_elt = -torch.log(1 - torch.sigmoid(neg_score) - 1e-8)               # [B,k]
    mask_e = mask.unsqueeze(1).float()                                      # [B,1]
    denom = mask_e.sum() * per_elt.shape[1] + 1e-8
    return (per_elt * mask_e).sum() / denom

def get_critique_neg_loss_gated(user_embed, neg_embed, gate):
    # In-training CF-gated neg loss: per-neg weighted push-down.
    # gate [B,k] = user-level mask x per-neg CF gating weight.
    neg_score = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=2)        # [B,k]
    per_elt = -torch.log(1 - torch.sigmoid(neg_score) - 1e-8)               # [B,k]
    denom = gate.sum() + 1e-8                                               # total valid weight
    return (per_elt * gate).sum() / denom

def get_critique_pos_loss_masked(user_embed, pos_embed, mask):
    # Masked positive critique loss (pulls up pos items): -log(sigmoid(user.pos)).
    pos_score = torch.sum(user_embed.unsqueeze(1) * pos_embed, dim=2)        # [B,k]
    per_elt = -torch.log(torch.sigmoid(pos_score) + 1e-8)                   # [B,k]
    mask_e = mask.unsqueeze(1).float()                                      # [B,1]
    denom = mask_e.sum() * per_elt.shape[1] + 1e-8
    return (per_elt * mask_e).sum() / denom

def get_prd_pos_loss_masked(user_embed, pos_pool_embed, pos_scores, pos_mask, tau=0.05, gap_power=1.0):
    # Preference-ranking distillation on positive candidates (Bradley-Terry MLE).
    # user_embed [B,d], pos_pool_embed [B,P,d], pos_scores [B,P] (padding=0), pos_mask [B] (1 = pos user).
    # Pairs with |s_i - s_j| > tau are counted; padding excluded.
    phi = torch.sum(user_embed.unsqueeze(1) * pos_pool_embed, dim=2)            # [B,P] CF affinity
    sdiff = pos_scores.unsqueeze(2) - pos_scores.unsqueeze(1)                   # [B,P,P] s_i - s_j
    phi_diff = phi.unsqueeze(2) - phi.unsqueeze(1)                              # [B,P,P] phi_i - phi_j
    real = (pos_scores > 0).float()                                             # [B,P] non-padding
    valid = (sdiff > tau).float() * real.unsqueeze(2) * real.unsqueeze(1)       # [B,P,P] valid ordered pairs
    weight = sdiff.clamp(min=0) ** gap_power                                    # pair confidence weight
    loss_elt = weight * (-torch.log(torch.sigmoid(phi_diff) + 1e-8))            # [B,P,P] -log sigma(phi_i-phi_j)
    mask_e = pos_mask.unsqueeze(1).unsqueeze(2).float()                         # [B,1,1] pos users only
    denom = (valid * mask_e).sum() + 1e-8
    return (loss_elt * valid * mask_e).sum() / denom

def get_critique_InfoNCE_loss(user_embed, pos_embed, cl_embed):

    user_embed_expand = user_embed.unsqueeze(1)

    pos_score = torch.sum(user_embed * pos_embed, dim=1)
    cl_score = torch.sum(user_embed_expand * cl_embed, dim=2)
    total_score = torch.cat([pos_score.unsqueeze(1), cl_score], dim=1)

    pos_score = torch.exp(pos_score)
    total_score = torch.exp(total_score).sum(dim=1)

    infoNCE_loss = -torch.mean(torch.log(pos_score / total_score + 10e-6))

    return infoNCE_loss

def get_reg_loss(*embeddings):
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])
    return reg_loss

def get_InfoNCE_loss(embedding1, embedding2, temperature):
    embedding1 = torch.nn.functional.normalize(embedding1)
    embedding2 = torch.nn.functional.normalize(embedding2)

    pos_score = (embedding1 * embedding2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    total_score = torch.matmul(embedding1, embedding2.transpose(0, 1))
    total_score = torch.exp(total_score / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / total_score + 10e-6)
    return torch.mean(cl_loss)


def get_neighbor_aggregate_loss(embedding1, embedding2, tau):
    embedding1 = torch.nn.functional.normalize(embedding1)
    embedding2 = torch.nn.functional.normalize(embedding2)

    pos_score = (embedding1 * embedding2).sum(dim=-1)
    pos_score = torch.exp(pos_score / tau)

    # pairwise similarity matrix (batch_size x batch_size)
    total_score = torch.matmul(embedding1, embedding2.transpose(0, 1)) + torch.matmul(embedding1, embedding1.transpose(0, 1))
    total_score = torch.exp(total_score / tau).sum(dim=1)

    # InfoNCE: -log(positive / total); 10e-6 for numerical stability
    na_loss = -torch.log(pos_score / total_score + 10e-6)
    return torch.mean(na_loss)


    return torch.mean(cl_loss)
