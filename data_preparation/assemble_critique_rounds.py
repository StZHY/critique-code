"""Assemble LLM evidence into per-round negative-only critique files (test_{r}_neg.txt + _scores) consumed by critique_data_loader."""
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
                continue
            bid = arecs[r]["book_id"]
            whitelist = pool_ids.get(u, set())
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
