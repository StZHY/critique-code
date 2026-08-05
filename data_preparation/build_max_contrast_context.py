"""Build per-anchor critique context with the MAXIMUM-CONTRAST comparison pair from the TEST holdout; writes prepare_for_LLM/user_test_critique.json."""
import json
import os
import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ, "dataset/amazon-book-82p")
INTER = os.path.join(DATA_DIR, "interactions.jsonl")
TITLE_FILE = os.environ.get("TITLE_FILE", "processed_Amazonbooks.dat")
PTH = os.path.join(PROJ, "model_save/best_model_LightCCF.pth")
OUT_DIR = os.path.join(PROJ, "prepare_for_LLM")
OUT = os.path.join(OUT_DIR, "user_test_critique.json")


def load_title_map(path):
    id2title = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                try:
                    id2title[int(parts[0])] = parts[2]
                except ValueError:
                    continue
    return id2title


def load_interactions(path):
    """Bucket each user: test_like/test_dislike carry review (anchor candidates); all_like/all_dislike carry id+time (comparison pool)."""
    users = defaultdict(lambda: {"test_like": [], "test_dislike": [],
                                 "all_like": [], "all_dislike": []})
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            u = r["user_id"]; it = r["item_id"]; ov = r["overall"]
            ph = r["phase"]; rt = r.get("reviewText") or ""; t = r.get("unixReviewTime", 0)
            if ov >= 4:
                users[u]["all_like"].append((it, t))
                if ph == "test":
                    users[u]["test_like"].append((it, ov, rt, t))
            elif ov <= 2:
                users[u]["all_dislike"].append((it, t))
                if ph == "test":
                    users[u]["test_dislike"].append((it, ov, rt, t))
    return users


def load_item_embedding(pth_path):
    sd = torch.load(pth_path, map_location="cpu")
    return sd["item_embedding.weight"].numpy()


def find_best_pair(anchor_vec, pool_ids, pool_mat, cap=80):
    """Pick 2 pool items that MAXIMIZE the triangle perimeter S=d(a,pi)+d(a,pj)+d(pi,pj) (sharpest decision boundary)."""
    if len(pool_ids) < 2:
        return None
    d_anc = np.linalg.norm(pool_mat - anchor_vec, axis=1)
    idxs = np.argsort(-d_anc)[:cap] if len(pool_ids) > cap else np.arange(len(pool_ids))
    if len(idxs) < 2:
        return None
    sub_mat = pool_mat[idxs]
    sub_ids = [pool_ids[i] for i in idxs]
    d_anc_sub = d_anc[idxs]
    d_pp = np.linalg.norm(sub_mat[:, None, :] - sub_mat[None, :, :], axis=2)
    best_S = -float("inf"); best = None
    for i, j in combinations(range(len(idxs)), 2):
        S = d_anc_sub[i] + d_anc_sub[j] + d_pp[i][j]
        if S > best_S:
            best_S = S; best = (sub_ids[i], sub_ids[j])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_rounds", type=int, default=5)
    ap.add_argument("--cap", type=int, default=80)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--title_file", default=TITLE_FILE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pth", default=PTH)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    inter = os.path.join(args.data_dir, "interactions.jsonl")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"data_dir={args.data_dir} | pth={args.pth} | out={args.out}")
    print("loading title map...")
    id2title = load_title_map(args.title_file)
    print(f"loading interactions ({os.path.basename(args.data_dir)})...")
    users = load_interactions(inter)
    print(f"  users: {len(users)}")
    print("loading backbone item embedding...")
    item_emb = load_item_embedding(args.pth)
    print(f"  item_emb shape: {item_emb.shape}")

    result = {}
    n_users = 0
    n_anchors = 0
    n_like = 0
    n_dislike = 0
    n_skip = 0
    cnt_round = defaultdict(int)

    keys = sorted(users.keys())
    if args.limit:
        keys = keys[:args.limit]
        print(f"  limit={args.limit}: processing first {len(keys)} users")
    for u in keys:
        d = users[u]
        cand = []
        for (it, ov, rt, t) in d["test_like"]:
            cand.append((t, "like", it, ov, rt))
        for (it, ov, rt, t) in d["test_dislike"]:
            cand.append((t, "dislike", it, ov, rt))
        cand.sort(key=lambda x: x[0])
        anchors = cand[:args.num_rounds]
        if not anchors:
            n_skip += 1
            continue

        anchor_ids = {a[2] for a in anchors}
        like_pool = [it for (it, _t) in d["all_like"] if it not in anchor_ids and it < item_emb.shape[0]]
        like_pool = list(dict.fromkeys(like_pool))
        dislike_pool = [it for (it, _t) in d["all_dislike"] if it not in anchor_ids and it < item_emb.shape[0]]
        dislike_pool = list(dict.fromkeys(dislike_pool))
        like_mat = item_emb[like_pool] if like_pool else None
        dislike_mat = item_emb[dislike_pool] if dislike_pool else None

        recs = []
        for (_t, typ, it, ov, rt) in anchors:
            anchor_vec = item_emb[it]
            if typ == "dislike":
                pool_ids, pool_mat = like_pool, like_mat
            else:
                pool_ids, pool_mat = dislike_pool, dislike_mat
            pair = find_best_pair(anchor_vec, pool_ids, pool_mat, args.cap) if pool_ids is not None else None
            if pair is None:
                cids, citems = [], []
            else:
                cids = list(pair)
                citems = [id2title.get(c, "") for c in cids]
            recs.append({
                "type": typ,
                "book_id": it,
                "title": id2title.get(it, ""),
                "overall": ov,
                "reviewText": rt,
                "unixReviewTime": _t,
                "contrast_item_ids": cids,
                "contrast_items": citems,
            })
            if typ == "like":
                n_like += 1
            else:
                n_dislike += 1
        if recs:
            result[str(u)] = recs
            n_users += 1
            n_anchors += len(recs)
            for k in range(1, len(recs) + 1):
                cnt_round[k] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== done ===")
    print(f"users with test critique anchors: {n_users} / {len(users)} (skipped {n_skip})")
    print(f"total anchors: {n_anchors} (like {n_like} / dislike {n_dislike})")
    print(f"anchors-per-user distribution:")
    ac = sorted(len(v) for v in result.values())
    if ac:
        print(f"  min={ac[0]} median={ac[len(ac)//2]} mean={sum(ac)/len(ac):.1f} max={ac[-1]}")
    for k in range(1, args.num_rounds + 1):
        print(f"  users with >={k} anchors: {cnt_round.get(k,0)} (rounds 0..{k-1})")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
