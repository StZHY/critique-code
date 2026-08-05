"""Loss functions for Critique_rllm (BPR/CL plus the residual-LLM regression and pairwise terms)."""
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
    neg_score = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=2)
    per_elt = -torch.log(1 - torch.sigmoid(neg_score) - 1e-8)
    mask_e = mask.unsqueeze(1).float()
    denom = mask_e.sum() * per_elt.shape[1] + 1e-8
    return (per_elt * mask_e).sum() / denom

def get_critique_neg_loss_gated(user_embed, neg_embed, gate):
    neg_score = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=2)
    per_elt = -torch.log(1 - torch.sigmoid(neg_score) - 1e-8)
    denom = gate.sum() + 1e-8
    return (per_elt * gate).sum() / denom

def get_critique_neg_loss_gateaware(user_embed, neg_embed, gate_pen, mask=None, gate=None):
    """Gate-aware neg BPR (P, CF x LLM co-action): subtract the LLM gate penalty gate_pen from neg_score
    so the residual gradient is suppressed on FP/dislike items the inference gate already handles."""
    neg_score = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=2) - gate_pen   # [B,K]
    per_elt = -torch.log(1 - torch.sigmoid(neg_score) - 1e-8)
    if gate is not None:
        denom = gate.sum() + 1e-8
        return (per_elt * gate).sum() / denom
    elif mask is not None:
        mask_e = mask.unsqueeze(1).float()
        denom = mask_e.sum() * per_elt.shape[1] + 1e-8
        return (per_elt * mask_e).sum() / denom
    else:
        return torch.mean(per_elt)


def get_semantic_offset_bpr_loss(user_embed, pos_embed, neg_embed, q_neg, tau=1.0, neg_mask=None):
    """Semantic-Offset L_cri (pairwise joint-margin BPR): CL anchor (pos) vs each critique dislike (neg)
    on the joint score s' = s_delta + q, with L = -log sigma(((s_pos - s_neg) - q_neg) / tau).
    q_neg = -M*sigma(gamma*ell) <= 0 (FP items only, no grad) gives gated pairs a positive margin."""
    s_pos = torch.sum(user_embed * pos_embed, dim=1, keepdim=True)          # [B,1]
    s_neg = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=2)           # [B,Kn]
    delta = (s_pos - s_neg) - q_neg                                         # [B,Kn]
    per_pair = -torch.log(torch.sigmoid(delta / tau) + 1e-8)               # [B,Kn]
    if neg_mask is not None:
        m = neg_mask.float().unsqueeze(1)                                   # [B,1]
        denom = m.sum() * per_pair.shape[1] + 1e-8
        return (per_pair * m).sum() / denom
    return torch.mean(per_pair)


def get_critique_pos_loss_masked(user_embed, pos_embed, mask):
    pos_score = torch.sum(user_embed.unsqueeze(1) * pos_embed, dim=2)
    per_elt = -torch.log(torch.sigmoid(pos_score) + 1e-8)
    mask_e = mask.unsqueeze(1).float()
    denom = mask_e.sum() * per_elt.shape[1] + 1e-8
    return (per_elt * mask_e).sum() / denom

def get_prd_pos_loss_masked(user_embed, pos_pool_embed, pos_scores, pos_mask, tau=0.05, gap_power=1.0):
    phi = torch.sum(user_embed.unsqueeze(1) * pos_pool_embed, dim=2)
    sdiff = pos_scores.unsqueeze(2) - pos_scores.unsqueeze(1)
    phi_diff = phi.unsqueeze(2) - phi.unsqueeze(1)
    real = (pos_scores > 0).float()
    valid = (sdiff > tau).float() * real.unsqueeze(2) * real.unsqueeze(1)
    weight = sdiff.clamp(min=0) ** gap_power
    loss_elt = weight * (-torch.log(torch.sigmoid(phi_diff) + 1e-8))
    mask_e = pos_mask.unsqueeze(1).unsqueeze(2).float()
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


def get_cer_loss(user_embed, pos_embed, crit_neg_embed, rand_neg_embed, tau=1.0):
    """Three-bucket InfoNCE (v2 Eq(18)): explicit B+ (history anchors), B- (past critique), Bref (random)."""
    pos_score = torch.sum(user_embed * pos_embed, dim=1)                          # [B]
    crit_score = torch.sum(user_embed.unsqueeze(1) * crit_neg_embed, dim=2)       # [B,|B-|]
    rand_score = torch.sum(user_embed.unsqueeze(1) * rand_neg_embed, dim=2)       # [B,|Bref|]
    num = torch.exp(pos_score / tau)                                              # [B]
    den = num + torch.exp(crit_score / tau).sum(dim=1) + torch.exp(rand_score / tau).sum(dim=1)
    return -torch.mean(torch.log(num / den + 1e-8))

def get_cer_margin_loss(user_embed, pos_embed, rand_neg_embed, cri_neg_embed, cri_neg_g,
                        margin=0.5, tau=0.1):
    """s_L-margin CER: fold the LLM dislike ordering into the CER structure by adding an additive
    margin m*g_n (g_n = sigma(gamma*ell_n)) to the dislike bucket B- logits in the InfoNCE denominator."""
    s_pos = torch.sum(user_embed * pos_embed, dim=1)                              # [B]
    s_rand = torch.sum(user_embed.unsqueeze(1) * rand_neg_embed, dim=2)           # [B,Kr]
    s_cri = torch.sum(user_embed.unsqueeze(1) * cri_neg_embed, dim=2)             # [B,Kc]
    valid_cri = (cri_neg_g > 0).float()                                            # [B,Kc] padding mask
    num = torch.exp(s_pos / tau)                                                  # [B]
    den = num + torch.exp(s_rand / tau).sum(dim=1) \
              + (torch.exp((s_cri + margin * cri_neg_g) / tau) * valid_cri).sum(dim=1)
    return -torch.mean(torch.log(num / den + 1e-8))


def get_ccr_loss(user_embed, pos_embed, rand_neg_embed, cri_neg_embed, cri_neg_q, tau=1.0):
    """Unified CCR (v0.5.0 §4.3): set-wise Z+/Z- contrast on the joint score z = s_delta + q (q no grad).
    Unlike margin-CER, the dislike bucket logit is *decreased* by q<0 so gate-corrected dislikes
    contribute less to Z- (gradient redistribution, §5.1)."""
    s_pos = torch.sum(user_embed * pos_embed, dim=1)                              # [B]
    s_rand = torch.sum(user_embed.unsqueeze(1) * rand_neg_embed, dim=2)           # [B,Kr]
    s_cri = torch.sum(user_embed.unsqueeze(1) * cri_neg_embed, dim=2)             # [B,Kc]
    z_cri = s_cri + cri_neg_q                                                     # [B,Kc] joint score (q no grad)
    valid_cri = (cri_neg_q < 0).float()                                           # [B,Kc] padding mask
    num = torch.exp(s_pos / tau)                                                  # [B]
    den = num + torch.exp(s_rand / tau).sum(dim=1) \
              + (torch.exp(z_cri / tau) * valid_cri).sum(dim=1)
    return -torch.mean(torch.log(num / den + 1e-8))


def get_reg_loss(*embeddings):
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])
    return reg_loss


# residual-LLM terms: gradients flow only to res_embed (item embeddings are detached/frozen => zero D6 propagation).

def get_residual_llm_loss_reg(res_embed, pool_embed, pool_scores):
    """Form A (regression): fit the residual head to the per-user standardized sign-coded s_L."""
    score = torch.sum(res_embed.unsqueeze(1) * pool_embed, dim=2)       # [B,P]
    real = (pool_scores != 0).float()                                   # [B,P]
    cnt = real.sum(dim=1, keepdim=True)                                 # [B,1]
    mu = (pool_scores * real).sum(dim=1, keepdim=True) / (cnt + 1e-8)
    var = ((pool_scores - mu) ** 2 * real).sum(dim=1, keepdim=True) / (cnt + 1e-8)
    sigma = torch.sqrt(var) + 1e-6
    target = (pool_scores - mu) / sigma                                 # [B,P]
    se = (score - target) ** 2 * real                                   # [B,P]
    denom = real.sum() + 1e-8
    return se.sum() / denom


def get_residual_llm_loss_pair(res_embed, pool_embed, pool_scores, tau=0.05):
    """Form B (pairwise BPR): reproduce the LLM preference total order on the unified pool."""
    score = torch.sum(res_embed.unsqueeze(1) * pool_embed, dim=2)        # [B,P]
    real = (pool_scores != 0).float()                                    # [B,P]
    sdiff = pool_scores.unsqueeze(2) - pool_scores.unsqueeze(1)          # [B,P,P]
    score_diff = score.unsqueeze(2) - score.unsqueeze(1)                 # [B,P,P]
    valid = (sdiff > tau).float() * real.unsqueeze(2) * real.unsqueeze(1)  # [B,P,P]
    loss_elt = -torch.log(torch.sigmoid(score_diff) + 1e-8)              # [B,P,P]
    denom = valid.sum() + 1e-8
    return (loss_elt * valid).sum() / denom


def get_residual_llm_loss_pair_v2(score, pool_scores, beta=0.0, within_w=1.0, tau=0.05):
    """Form B-v2 (prediction-space + soft label + polarity stratification)."""
    real = (pool_scores != 0).float()                                     # [B,P]
    sdiff = pool_scores.unsqueeze(2) - pool_scores.unsqueeze(1)           # [B,P,P]
    score_diff = score.unsqueeze(2) - score.unsqueeze(1)                  # [B,P,P]
    p_ij = torch.sigmoid(beta * sdiff) if beta > 0 else (sdiff > 0).float()
    if within_w < 1.0:
        sign_i = torch.sign(pool_scores).unsqueeze(2)                     # [B,P,1]
        sign_j = torch.sign(pool_scores).unsqueeze(1)                     # [B,1,P]
        cross = (sign_i * sign_j < 0).float()                             # [B,P,P]
        w_ij = cross + (1.0 - cross) * within_w                           # cross=1 / same=within_w
    else:
        w_ij = 1.0
    valid = real.unsqueeze(2) * real.unsqueeze(1) * (sdiff.abs() > tau).float() * w_ij
    q = torch.sigmoid(score_diff)
    loss_elt = -(p_ij * torch.log(q + 1e-8) + (1.0 - p_ij) * torch.log(1.0 - q + 1e-8))
    denom = valid.sum() + 1e-8
    return (loss_elt * valid).sum() / denom


def get_residual_llm_loss_pair_v2_polarity(score_prop, score_neg, pool_scores,
                                           beta=0.0, within_w=1.0, tau=0.05):
    """Form B-v2 polarity-aware: pos members use the propagated score_prop, neg members the
    non-propagated score_neg (npc_prop_neg routing) within the unified pair loss."""
    real = (pool_scores != 0).float()                                     # [B,P]
    pos_m = (pool_scores > 0).float()
    neg_m = (pool_scores < 0).float()
    score = pos_m * score_prop + neg_m * score_neg                        # [B,P]
    sdiff = pool_scores.unsqueeze(2) - pool_scores.unsqueeze(1)           # [B,P,P]
    score_diff = score.unsqueeze(2) - score.unsqueeze(1)                  # [B,P,P]
    p_ij = torch.sigmoid(beta * sdiff) if beta > 0 else (sdiff > 0).float()
    if within_w < 1.0:
        sign_i = torch.sign(pool_scores).unsqueeze(2)
        sign_j = torch.sign(pool_scores).unsqueeze(1)
        cross = (sign_i * sign_j < 0).float()
        w_ij = cross + (1.0 - cross) * within_w
    else:
        w_ij = 1.0
    valid = real.unsqueeze(2) * real.unsqueeze(1) * (sdiff.abs() > tau).float() * w_ij
    q = torch.sigmoid(score_diff)
    loss_elt = -(p_ij * torch.log(q + 1e-8) + (1.0 - p_ij) * torch.log(1.0 - q + 1e-8))
    denom = valid.sum() + 1e-8
    return (loss_elt * valid).sum() / denom


def get_residual_orth_loss(res_embed, anchor_embed):
    """(a) Orthogonal protection: L_prot = mean(<delta_u, phi_pos_bar>^2) drives the residual's
    projection on the train-positive-mean direction to zero."""
    proj = torch.sum(res_embed * anchor_embed, dim=1)        # [B]
    return torch.mean(proj ** 2)
