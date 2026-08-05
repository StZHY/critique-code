# -*- coding: utf-8 -*-
"""Assemble the LLM evidence into per-round negative-only critique files.

This is the main (negative-only) critique protocol. Per user, round r is the pool anchored on
the user's r-th top dislike anchor; the anchor is forced at index 0 and the LLM-generated
dislike candidates (already produced over the constrained pool) follow, deduped and
whitelist-filtered.

Major steps:
  1. Read the dislike anchors, the LLM candidate suggestions and the pool whitelist.
  2. For each round r in 0..num_rounds-1 and each user with an r-th anchor, build
     items = [anchor] + filtered candidates and scores = [1.0] + candidate scores.
  3. Write dataset/.../critique_round_negonly_top20/test_{r}_neg.txt and
     test_{r}_neg_scores.txt, the files consumed by utility/critique_data_loader.py.

Input files (under prepare_for_LLM/):
  top20_neg_anchors_gcn2.json   {user: [dislike anchor {book_id, rank, ...}]}
  batch_llm_neg_gcn2_full.json  {user: {book_id: ["id::title::score"]}}
  top_recommendations_gcn2_100.json  {user: ["id::title"]}  pool whitelist (defensive filter)
Line format: "{user} {item0} {item1} ..." / "{user} {score0} {score1} ..."
"""
import os
import json
import argparse
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHORS = os.path.join(PROJ, "prepare_for_LLM/top20_neg_anchors_gcn2.json")
CANDS = os.path.join(PROJ, "prepare_for_LLM/batch_llm_neg_gcn2_full.json")
POOL = os.path.join(PROJ, "prepare_for_LLM/top_recommendations_gcn2_100.json")
OUT_DIR = os.path.join(PROJ, "dataset/amazon-book-82p-rand/critique_round_negonly_top20")
NUM_ROUNDS = 5


def parse_cand(s):
    """'id::title::score' -> (id:int, score:float); skip on failure."""
    try:
        parts = s.split("::")
        if len(parts) >= 3:
            return int(parts[0]), float(parts[-1])
    except (ValueError, TypeError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=ANCHORS)
    ap.add_argument("--cands", default=CANDS)
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    args = ap.parse_args()

    print(f"read anchors: {args.anchors}")
    anchors = json.load(open(args.anchors, encoding="utf-8"))
    print(f"  {len(anchors)} users, {sum(len(v) for v in anchors.values())} anchors")
    print(f"read candidates: {args.cands}")
    cands = json.load(open(args.cands, encoding="utf-8"))
    print(f"  {len(cands)} users")
    print(f"read pool whitelist: {args.pool}")
    pool = json.load(open(args.pool, encoding="utf-8"))

    # pool whitelist (defensive: candidates must be inside the pool)
    pool_ids = {}
    for u, lst in pool.items():
        pool_ids[str(u)] = set(int(e.split("::")[0]) for e in lst if "::" in e)

    os.makedirs(args.out_dir, exist_ok=True)
    round_stats = defaultdict(lambda: {"users": 0, "items": 0, "no_cand": 0})

    for r in range(args.num_rounds):
        id_lines, sco_lines = [], []
        for u in sorted(anchors.keys(), key=lambda x: int(x)):
            arecs = anchors[u]
            if r >= len(arecs):
                continue  # this user has no r-th anchor -> no negatives this round
            bid = arecs[r]["book_id"]
            whitelist = pool_ids.get(u, set())
            # candidates: parse + dedup + exclude anchor + pool-whitelist filter
            seen = {bid}
            cand_items = []
            cand_scores = []
            raw = cands.get(u, {}).get(str(bid), []) or []
            for s in raw:
                pc = parse_cand(s)
                if pc is None:
                    continue
                cid, cscore = pc
                if cid in seen:
                    continue
                if whitelist and cid not in whitelist:
                    continue
                seen.add(cid)
                cand_items.append(cid)
                cand_scores.append(cscore)
            # anchor at idx0 (force_neg_anchor), score=1.0 (forced; score not used in importance sampling)
            items = [bid] + cand_items
            scores = [1.0] + cand_scores
            if not cand_items:
                round_stats[r]["no_cand"] += 1
            id_lines.append(f"{u} " + " ".join(str(it) for it in items))
            sco_lines.append(f"{u} " + " ".join(f"{s:.6f}" for s in scores))
            round_stats[r]["users"] += 1
            round_stats[r]["items"] += len(items)
        with open(os.path.join(args.out_dir, f"test_{r}_neg.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(id_lines) + ("\n" if id_lines else ""))
        with open(os.path.join(args.out_dir, f"test_{r}_neg_scores.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sco_lines) + ("\n" if sco_lines else ""))

    print("\n=== done ===")
    tot_u = sum(s["users"] for s in round_stats.values())
    for r in range(args.num_rounds):
        s = round_stats[r]
        mean_items = s["items"] / max(s["users"], 1)
        print(f"  round {r}: {s['users']} users, {s['items']} items (mean {mean_items:.1f}/user), "
              f"no-cand (anchor-only) {s['no_cand']}")
    print(f"  total user-rounds: {tot_u}")
    print(f"output: {args.out_dir}/test_{{0..{args.num_rounds-1}}}_neg.txt (+_scores)")


if __name__ == "__main__":
    main()
