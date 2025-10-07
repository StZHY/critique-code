import numpy as np
import torch
import torch.nn.functional as F
import utility.tools as tools



def simple_sample_bpr_neg_items(model, args, dataset, device):
    
    num_users = len(dataset.neg_train_dict)
    bpr_neg_items = torch.zeros((num_users, args.num_bpr_neg), dtype=torch.long, device=device)

    sampled_neg_items = {}
    
    model.eval()
    with torch.no_grad():
        for user_id, data in dataset.neg_train_dict.items():
            neg_ids = data['neg_ids']
            neg_scores = data['neg_scores']

            if not neg_ids:
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

            sampled_indices = torch.multinomial(
                input=ws,
                num_samples=args.num_bpr_neg,
                replacement=True
            )
            
            sampled_ids = neg_ids_tensor[sampled_indices]
            bpr_neg_items[user_id] = sampled_ids
            
            sampled_neg_items[user_id] = sampled_ids.tolist()

    return bpr_neg_items, sampled_neg_items

def weight_sample_bpr_neg_items(model, args, dataset, device):
    num_users = len(dataset.neg_train_dict)
    bpr_neg_items = torch.zeros((num_users, args.num_bpr_neg), dtype=torch.long, device=device)
    sampled_neg_items = {}

    model.eval()
    with torch.no_grad():
        for user_id, data in dataset.neg_train_dict.items():
            neg_ids = data['neg_ids']
            neg_scores = data['neg_scores']

            if not neg_ids:
                continue
            
            neg_ids_tensor = torch.tensor(neg_ids, dtype=torch.long, device=device)
            neg_scores_tensor = torch.tensor(neg_scores, dtype=torch.float, device=device)
            
            user_emb = model.user_embedding(torch.tensor([user_id], dtype=torch.long, device=device))
            item_embs = model.item_embedding(neg_ids_tensor)
            
            user_emb = torch.nan_to_num(user_emb, nan=0.0)
            item_embs = torch.nan_to_num(item_embs, nan=0.0)
            
            S_cf = torch.sum(user_emb * item_embs, dim=1)
            exponent_term = args.lambda_cf * S_cf + (1- args.lambda_cf) * neg_scores_tensor

            ws = torch.softmax(exponent_term, dim=0)
            if torch.sum(ws) == 0:
                ws = torch.ones_like(ws)

            sampled_indices = torch.multinomial(
                input=ws,
                num_samples=args.num_bpr_neg,
                replacement=True
            )

            sampled_ids = neg_ids_tensor[sampled_indices]
            bpr_neg_items[user_id] = sampled_ids

            sampled_neg_items[user_id] = sampled_ids.tolist()

    return bpr_neg_items, sampled_neg_items

def sample_cl_neg_items(args, dataset, device, history_bpr_neg):
    
    num_users = len(dataset.neg_train_dict)
    num_items = dataset.num_items
    num_cl_neg = int(args.num_cl_neg)
    
    num_rand_neg = int(num_cl_neg * 99/ 100)
    num_bpr_neg = num_cl_neg - num_rand_neg
    
    all_items = torch.arange(num_items).to(device)
    cl_neg_items = torch.zeros((num_users, num_cl_neg), dtype=torch.long).to(device)
    
    for user, data in dataset.neg_train_dict.items():
            
        user_history_set = set(dataset.train_dict.get(user, []))
            
        all_items_list = all_items.tolist()
        non_history_items = [item for item in all_items_list if item not in user_history_set]
        if not non_history_items:
            rand_sampled_items = []
        else:
            rand_sampled_items = np.random.choice(non_history_items, size=num_rand_neg, replace=True)
            
        bpr_history = history_bpr_neg.get(user, [])
        if not bpr_history:
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