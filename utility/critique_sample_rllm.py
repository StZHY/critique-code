"""Negative/positive sampling utilities for critique_trainer_rllm, incl. the unified sign-coded critique pool."""
import numpy as np
import torch
import torch.nn.functional as F
import utility.tools as tools


_CF_TABLE_CACHE = {}


def _load_cf_table(path):
    if path not in _CF_TABLE_CACHE:
        import json
        _CF_TABLE_CACHE[path] = json.load(open(path, encoding="utf-8"))
    return _CF_TABLE_CACHE[path]


def _apply_gate(cf_k, pmin, pmax, args):
    K = cf_k.shape[0]
    span = (pmax - pmin) + 1e-9
    cn = (cf_k - pmin) / span
    if getattr(args, 'neg_gate_invert', False):
        cn = 1.0 - cn
    mode = args.neg_gate_mode
    if mode == 'linear':
        w = cn
    elif mode == 'power':
        w = cn.clamp(min=0.0) ** args.neg_gate_gamma
    elif mode == 'softmax':
        w = torch.softmax(args.neg_gate_gamma * cn, dim=0) * K
    elif mode == 'hard_topk':
        topk = min(int(args.neg_gate_topk), K)
        w = torch.zeros(K, device=cf_k.device)
        if topk > 0:
            _, idx = torch.topk(cn, topk)
            w[idx] = 1.0
    elif mode == 'hard_thresh':
        w = (cn > 0.5).float()
    else:
        w = torch.ones(K, device=cf_k.device)
    return w


def compute_neg_gate(args, bpr_neg_items, device):
    if args.neg_gate_mode == 'none' or not getattr(args, 'neg_gate_cftable', None):
        return None
    table = _load_cf_table(args.neg_gate_cftable)
    num_users, K = bpr_neg_items.shape
    gate = torch.zeros((num_users, K), device=device)
    bpr_cpu = bpr_neg_items.cpu().tolist()
    for u in range(num_users):
        row = table.get(str(u))
        if not row:
            continue
        pool_cfs = list(row.values())
        pmin = float(min(pool_cfs))
        pmax = float(max(pool_cfs))
        cf_k = torch.tensor([float(row.get(str(neg), pmin)) for neg in bpr_cpu[u]],
                            device=device, dtype=torch.float)
        gate[u] = _apply_gate(cf_k, pmin, pmax, args)
    return gate


def choose_cri_items(model, args, dataset, device):
    num_users = dataset.num_users
    bpr_neg_items = torch.zeros((num_users, 1), dtype=torch.long, device=device)
    sampled_neg_items = {}
    for user_id, data in dataset.neg_train_dict.items():
        neg_ids = data['neg_ids']
        neg_id_with_max_score = neg_ids[0]
        bpr_neg_items[user_id] = neg_id_with_max_score
        sampled_neg_items[user_id] = [neg_id_with_max_score] * args.num_bpr_neg
    return bpr_neg_items, sampled_neg_items


def simple_sample_bpr_neg_items(model, args, dataset, device):
    num_users = dataset.num_users
    bpr_neg_items = torch.zeros((num_users, args.num_bpr_neg), dtype=torch.long, device=device)
    sampled_neg_items = {}
    model.eval()
    with torch.no_grad():
        for user_id, data in dataset.neg_train_dict.items():
            neg_ids = data['neg_ids']
            neg_scores = data['neg_scores']
            if not neg_ids:
                print(f"警告: 用户 {user_id} 没有负样本，跳过采样。")
                continue
            neg_ids_tensor = torch.tensor(neg_ids, dtype=torch.long, device=device)
            neg_scores_tensor = torch.tensor(neg_scores, dtype=torch.float, device=device)
            user_emb = model.user_embedding(torch.tensor([user_id], dtype=torch.long, device=device))
            item_embs = model.item_embedding(neg_ids_tensor)
            if torch.isnan(user_emb).any() or torch.isnan(item_embs).any():
                user_emb = torch.nan_to_num(user_emb, nan=0.0)
                item_embs = torch.nan_to_num(item_embs, nan=0.0)
            S_cf = torch.sum(user_emb * item_embs, dim=1)
            S_cf_min = torch.min(S_cf)
            S_cf_max = torch.max(S_cf)
            S_cf_norm = (S_cf - S_cf_min) / (S_cf_max - S_cf_min + 1e-9)
            exp_rs = torch.exp(neg_scores_tensor)
            ws = (neg_scores_tensor * S_cf_norm) / (exp_rs + 1e-9)
            if torch.sum(ws) == 0:
                ws = torch.ones_like(ws)
            sampled_indices = torch.multinomial(input=ws, num_samples=args.num_bpr_neg, replacement=True)
            sampled_ids = neg_ids_tensor[sampled_indices]
            bpr_neg_items[user_id] = sampled_ids
            sampled_neg_items[user_id] = sampled_ids.tolist()
    return bpr_neg_items, sampled_neg_items

def weight_sample_bpr_neg_items(model, args, dataset, device):
    num_users = dataset.num_users
    bpr_neg_items = torch.zeros((num_users, args.num_bpr_neg), dtype=torch.long, device=device)
    sampled_neg_items = {}
    model.eval()
    with torch.no_grad():
        for user_id, data in dataset.neg_train_dict.items():
            neg_ids = data['neg_ids']
            neg_scores = data['neg_scores']
            if not neg_ids:
                print(f"警告: 用户 {user_id} 没有负样本，跳过采样。")
                continue
            neg_ids_tensor = torch.tensor(neg_ids, dtype=torch.long, device=device)
            neg_scores_tensor = torch.tensor(neg_scores, dtype=torch.float, device=device)
            user_emb = model.user_embedding(torch.tensor([user_id], dtype=torch.long, device=device))
            item_embs = model.item_embedding(neg_ids_tensor)
            user_emb = torch.nan_to_num(user_emb, nan=0.0)
            item_embs = torch.nan_to_num(item_embs, nan=0.0)
            S_cf = torch.sum(user_emb * item_embs, dim=1)
            exponent_term = args.lambda_cf * S_cf + (1 - args.lambda_cf) * neg_scores_tensor
            lambda_cross = float(getattr(args, 'lambda_cross', 0.0))
            if lambda_cross > 0.0:
                S_cf_n = (S_cf - S_cf.min()) / (S_cf.max() - S_cf.min() + 1e-9)
                S_l_n = (neg_scores_tensor - neg_scores_tensor.min()) / (neg_scores_tensor.max() - neg_scores_tensor.min() + 1e-9)
                exponent_term = exponent_term + lambda_cross * S_cf_n * S_l_n
            ws = torch.softmax(exponent_term, dim=0)
            if torch.sum(ws) == 0:
                ws = torch.ones_like(ws)
            K = args.num_bpr_neg
            if getattr(args, "force_neg_anchor", False) and len(neg_ids) >= 1:
                anchor = neg_ids_tensor[0]
                if len(neg_ids) == 1:
                    sampled_ids = anchor.repeat(K)
                else:
                    ws_cand = ws[1:]
                    if torch.sum(ws_cand) == 0:
                        ws_cand = torch.ones_like(ws_cand)
                    idx = torch.multinomial(input=ws_cand, num_samples=K - 1, replacement=True)
                    sampled_ids = torch.cat([anchor.unsqueeze(0), neg_ids_tensor[1:][idx]])
            else:
                sampled_indices = torch.multinomial(input=ws, num_samples=K, replacement=True)
                sampled_ids = neg_ids_tensor[sampled_indices]
            bpr_neg_items[user_id] = sampled_ids
            sampled_neg_items[user_id] = sampled_ids.tolist()
    return bpr_neg_items, sampled_neg_items

def sample_pos_items(model, args, dataset, device):
    num_users = dataset.num_users
    K = args.num_bpr_neg
    pos_items = torch.zeros((num_users, K), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for user_id, data in getattr(dataset, 'pos_train_dict', {}).items():
            pos_ids = data['pos_ids']
            pos_scores = data['pos_scores']
            if not pos_ids:
                continue
            ids_t = torch.tensor(pos_ids, dtype=torch.long, device=device)
            if len(pos_ids) == 1:
                pos_items[user_id] = ids_t[0].repeat(K)
            else:
                anchor = ids_t[0]
                cand = ids_t[1:]
                sc = torch.tensor(pos_scores[1:], dtype=torch.float, device=device)
                ws = torch.softmax(sc, dim=0)
                if torch.sum(ws) == 0:
                    ws = torch.ones_like(ws)
                idx = torch.multinomial(ws, num_samples=K - 1, replacement=True)
                pos_items[user_id] = torch.cat([anchor.unsqueeze(0), cand[idx]])
    return pos_items

def sample_pos_pool(args, dataset, device, max_pool=30):
    num_users = dataset.num_users
    pool_ids = torch.zeros((num_users, max_pool), dtype=torch.long, device=device)
    pool_scores = torch.zeros((num_users, max_pool), dtype=torch.float, device=device)
    for user_id, data in getattr(dataset, 'pos_train_dict', {}).items():
        pos_ids = data['pos_ids']
        pos_scores = data['pos_scores']
        if not pos_ids:
            continue
        n = min(len(pos_ids), max_pool)
        pool_ids[user_id, :n] = torch.tensor(pos_ids[:n], dtype=torch.long, device=device)
        pool_scores[user_id, :n] = torch.tensor(pos_scores[:n], dtype=torch.float, device=device)
    return pool_ids, pool_scores


def sample_critique_pool(args, dataset, device, max_pool=None):
    """Unified per-user critique pool (pos|neg) with sign-coded s_L (pos->+score / neg->-score).
    Merging pos_train_dict and neg_train_dict and negating the neg scores gives a single continuous
    preference signal in [-1,1]. padding = item 0 + score 0 (masked out in the loss via (scores!=0))."""
    if max_pool is None:
        max_pool = int(getattr(args, 'rllm_pool_size', 60))
    num_users = dataset.num_users
    pool_ids = torch.zeros((num_users, max_pool), dtype=torch.long, device=device)
    pool_scores = torch.zeros((num_users, max_pool), dtype=torch.float, device=device)
    pos_dict = getattr(dataset, 'pos_train_dict', {})
    neg_dict = getattr(dataset, 'neg_train_dict', {})
    user_set = set(pos_dict.keys()) | set(neg_dict.keys())
    for uid in user_set:
        ids, sL = [], []
        d = pos_dict.get(uid)
        if d and d.get('pos_ids'):
            ids.extend(d['pos_ids'])
            sL.extend([+float(s) for s in d['pos_scores'][:len(d['pos_ids'])]])
        d = neg_dict.get(uid)
        if d and d.get('neg_ids'):
            ids.extend(d['neg_ids'])
            sL.extend([-float(s) for s in d['neg_scores'][:len(d['neg_ids'])]])
        if not ids:
            continue
        n = min(len(ids), max_pool)
        pool_ids[uid, :n] = torch.tensor(ids[:n], dtype=torch.long, device=device)
        pool_scores[uid, :n] = torch.tensor(sL[:n], dtype=torch.float, device=device)
    return pool_ids, pool_scores

def sample_cri_neg_cl(args, dataset, device, Kc=None):
    """s_L-margin CER B- bucket: sample Kc LLM dislikes per user, return ids + g=sigma(gamma*ell).
    Only neg_train_dict dislikes are used; padding = item 0 + g=0 (masked out via (g>0) in the loss)."""
    if Kc is None:
        Kc = int(getattr(args, 'cer_cri_neg_k', 20))
    gamma = float(getattr(args, 'cer_gamma', 1.0))
    num_users = dataset.num_users
    cri_neg_ids = torch.zeros((num_users, Kc), dtype=torch.long, device=device)
    cri_neg_g = torch.zeros((num_users, Kc), dtype=torch.float, device=device)
    for uid, data in dataset.neg_train_dict.items():
        neg_ids = data.get('neg_ids', [])
        neg_scores = data.get('neg_scores', [])
        if not neg_ids:
            continue
        n = min(len(neg_ids), Kc)
        ids_t = torch.tensor(neg_ids[:n], dtype=torch.long, device=device)
        sc_t = torch.tensor([float(s) for s in neg_scores[:n]], dtype=torch.float, device=device)
        cri_neg_ids[uid, :n] = ids_t
        cri_neg_g[uid, :n] = torch.sigmoid(gamma * sc_t)
    return cri_neg_ids, cri_neg_g


def _sample_cl_neg_items_legacy(args, dataset, device, history_bpr_neg):
    """Legacy (byte-identical): cl_negative = 99% random (Bref) + 1% bpr_history (B-), merged into one tensor."""
    num_users = dataset.num_users
    num_items = dataset.num_items
    num_cl_neg = int(args.num_cl_neg)
    num_rand_neg = int(num_cl_neg * 99 / 100)
    num_bpr_neg = num_cl_neg - num_rand_neg
    all_items = torch.arange(num_items).to(device)
    cl_neg_items = torch.zeros((num_users, num_cl_neg), dtype=torch.long).to(device)
    pos_dict = getattr(dataset, 'pos_train_dict', {})
    user_set = set(dataset.neg_train_dict.keys()) | set(pos_dict.keys())
    for user in user_set:
        user_history_set = set(dataset.train_dict.get(user, []))
        all_items_list = all_items.tolist()
        non_history_items = [item for item in all_items_list if item not in user_history_set]
        if not non_history_items:
            print(f"警告: 用户 {user} 没有可供随机采样的非历史电影，跳过随机采样。")
            rand_sampled_items = []
        else:
            rand_sampled_items = np.random.choice(non_history_items, size=num_rand_neg, replace=True)
        bpr_history = history_bpr_neg.get(user, [])
        if not bpr_history:
            print(f"警告: 用户 {user} 没有BPR历史样本，跳过BPR采样。")
            bpr_sampled_items = []
        else:
            weights = 1 / (np.arange(len(bpr_history)) + 1)
            weights = weights / np.sum(weights)
            bpr_sampled_items = np.random.choice(bpr_history, size=num_bpr_neg, replace=True, p=weights)
        combined_samples = list(rand_sampled_items) + list(bpr_sampled_items)
        if len(combined_samples) < num_cl_neg:
            remaining_samples = np.random.choice(all_items.tolist(), size=(num_cl_neg - len(combined_samples)), replace=True)
            combined_samples.extend(list(remaining_samples))
        cl_neg_items[user] = torch.tensor(combined_samples, dtype=torch.long, device=device)
    return cl_neg_items


def sample_cl_neg_items(args, dataset, device, history_bpr_neg):
    """CER (v2 §2.6) contrastive negative sampling: Bref (random) | B- (past-critique = bpr_history).
    Default (both flags off) -> legacy (99% random + 1% bpr_history), byte-identical. D-2 three-bucket
    ablation (EXP-A05) is controlled here:
      --no_cer_random_ref      => B- only (drop random reference); backfill random if B- insufficient.
      --no_cer_critique_replay => Bref only (drop past-critique)."""
    no_ref = bool(getattr(args, 'no_cer_random_ref', False))
    no_crit = bool(getattr(args, 'no_cer_critique_replay', False))
    if not no_ref and not no_crit:
        return _sample_cl_neg_items_legacy(args, dataset, device, history_bpr_neg)

    num_users = dataset.num_users
    num_items = dataset.num_items
    num_cl_neg = int(args.num_cl_neg)
    all_items = torch.arange(num_items).to(device)
    all_items_list = all_items.tolist()
    cl_neg_items = torch.zeros((num_users, num_cl_neg), dtype=torch.long).to(device)
    pos_dict = getattr(dataset, 'pos_train_dict', {})
    user_set = set(dataset.neg_train_dict.keys()) | set(pos_dict.keys())

    # Target bucket quota: no_ref -> all B-; no_crit -> all Bref; both on -> degenerate, fall back to all random.
    want_bpr = (not no_crit) and no_ref       # past-critique only
    want_rand = (not no_ref) and no_crit      # random only
    if no_ref and no_crit:                    # both buckets off = degenerate
        want_rand, want_bpr = True, False

    for user in user_set:
        user_history_set = set(dataset.train_dict.get(user, []))
        non_history_items = [it for it in all_items_list if it not in user_history_set]
        bpr_history = history_bpr_neg.get(user, [])

        rand_sampled, bpr_sampled = [], []
        if want_rand:
            pool = non_history_items if non_history_items else all_items_list
            rand_sampled = list(np.random.choice(pool, size=num_cl_neg, replace=True))
        if want_bpr:
            if bpr_history:
                weights = 1 / (np.arange(len(bpr_history)) + 1)
                weights = weights / np.sum(weights)
                bpr_sampled = list(np.random.choice(bpr_history, size=num_cl_neg, replace=True, p=weights))
            else:
                # User has no past-critique (e.g. round 0): backfill random to avoid InfoNCE having no negatives.
                pool = non_history_items if non_history_items else all_items_list
                bpr_sampled = list(np.random.choice(pool, size=num_cl_neg, replace=True))

        combined = rand_sampled + bpr_sampled
        if len(combined) < num_cl_neg:   # fallback
            combined.extend(list(np.random.choice(all_items_list,
                               size=(num_cl_neg - len(combined)), replace=True)))
        cl_neg_items[user] = torch.tensor(combined[:num_cl_neg], dtype=torch.long, device=device)
    return cl_neg_items
