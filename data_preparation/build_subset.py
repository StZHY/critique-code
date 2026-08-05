"""Build the per-user 8:2 train/test subset from the original Amazon reviews (IID split, both-polarity test gate)."""
import json
import os
import random
from collections import defaultdict

SEED = 2025

DEFAULT_SRC = os.environ.get("SRC", "interactions_with_reviews.jsonl")
DEFAULT_AMZ_DIR = os.environ.get("AMZ_DIR", "amazon_original")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJ, "dataset")

SPLITS = (0.80, 0.20)
OUTDIR = os.path.join(DATASET_ROOT, "amazon-book-82p-rand")
MIN_TOTAL = 15
MIN_TR_POS = 2


def load_all(src):
    """Stream the jsonl and aggregate every interaction per original user id."""
    users = defaultdict(list)
    n = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            users[rec["user_id"]].append({
                "time": rec.get("unixReviewTime") or 0,
                "overall": rec.get("overall"),
                "item": rec["item_id"],
                "asin": rec.get("asin", ""),
                "reviewText": rec.get("reviewText", ""),
                "summary": rec.get("summary", ""),
            })
            n += 1
    print(f"  read {n} interactions, {len(users)} users")
    return users


def split_user(items, splits):
    """Random split into train/test (pure 8:2, no val; standard IID CF protocol)."""
    s = list(items)
    random.Random(SEED).shuffle(s)
    m = len(s)
    p_tr, _ = splits
    c1 = int(round(m * p_tr))
    c1 = max(1, min(m - 1, c1))
    return s[:c1], s[c1:]


def cnt(seq, lo, hi):
    return sum(1 for x in seq if x["overall"] is not None and lo <= x["overall"] <= hi)


def passes(total, tr, te):
    """Pure 8:2 gate: enough total/positives and test must hold both likes and dislikes."""
    if total < MIN_TOTAL:
        return False
    if cnt(tr, 4, 5) < MIN_TR_POS:
        return False
    if cnt(te, 4, 5) < 1:
        return False
    if cnt(te, 0, 2) < 1:
        return False
    return True


def build(src, amz_dir):
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"\n=== build pure 8:2 subset -> {OUTDIR} (split {SPLITS}) ===")

    users = load_all(src)

    kept = {}
    for uid, items in users.items():
        tr, te = split_user(items, SPLITS)
        if passes(len(items), tr, te):
            kept[uid] = (tr, te)
    print(f"  kept users: {len(kept)}")

    new_id = {uid: i for i, uid in enumerate(sorted(kept.keys()))}

    remap2rev = {}
    with open(os.path.join(amz_dir, "user_list.dat"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.split()
            if len(p) >= 2:
                try:
                    remap2rev[int(p[1])] = p[0]
                except ValueError:
                    pass

    items_seen = {}
    n_inter = {"train": 0, "test": 0, "all": 0}
    rating_dist = defaultdict(int)

    f_inter = open(os.path.join(OUTDIR, "interactions.jsonl"), "w", encoding="utf-8")
    for uid in sorted(kept.keys()):
        nid = new_id[uid]
        tr, te = kept[uid]
        for phase, seq in [("train", tr), ("test", te)]:
            for x in seq:
                items_seen[x["item"]] = x["asin"]
                rating_dist[x["overall"]] += 1
                n_inter["all"] += 1
                n_inter[phase] += 1
                f_inter.write(json.dumps({
                    "user_id": nid,
                    "item_id": x["item"],
                    "overall": x["overall"],
                    "reviewText": x["reviewText"],
                    "summary": x["summary"],
                    "unixReviewTime": x["time"],
                    "phase": phase,
                }, ensure_ascii=False) + "\n")
    f_inter.close()

    def write_split(fname, phase):
        path = os.path.join(OUTDIR, fname)
        idx = {"train": 0, "test": 1}[phase]
        with open(path, "w") as fout:
            for uid in sorted(kept.keys()):
                nid = new_id[uid]
                seq = kept[uid][idx]
                items4 = [str(x["item"]) for x in seq if x["overall"] is not None and x["overall"] >= 4]
                if items4:
                    fout.write(f"{nid} {' '.join(items4)}\n")
    write_split("train.txt", "train")
    write_split("test.txt", "test")

    with open(os.path.join(OUTDIR, "user_list.dat"), "w") as f:
        f.write("new_id\torg_user_id\treviewerID\n")
        for uid in sorted(kept.keys()):
            f.write(f"{new_id[uid]}\t{uid}\t{remap2rev.get(uid,'')}\n")

    with open(os.path.join(OUTDIR, "item_list.dat"), "w") as f:
        f.write("org_item_id\tasin\n")
        for iid in sorted(items_seen):
            f.write(f"{iid}\t{items_seen[iid]}\n")

    with open(os.path.join(OUTDIR, "subset_report.txt"), "w") as f:
        def w(*a):
            print(*a)
            print(*a, file=f)
        w(f"=== pure 8:2 subset report (amazon-book-82p) ===")
        w(f"split(train:test) = {SPLITS} (no val)")
        w(f"critique source = test (late holdout)")
        w(f"early-stop signal = train ndcg@5 convergence (test monitors overfit)")
        w(f"gate: total>={MIN_TOTAL} & tr_pos>={MIN_TR_POS} & te_pos>=1 & te_neg>=1")
        w(f"kept users: {len(kept)}")
        w(f"items (subset-used): {len(items_seen)} (item id kept original, max={max(items_seen)})")
        w(f"interactions: total={n_inter['all']} train={n_inter['train']} test={n_inter['test']}")
        w(f"rating dist: {dict(sorted(rating_dist.items(), key=lambda x:(x[0] is None, x[0])))}")
    print(f"  done: {OUTDIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--amz_dir", default=DEFAULT_AMZ_DIR)
    a = ap.parse_args()
    build(a.src, a.amz_dir)
