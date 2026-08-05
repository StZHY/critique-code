# -*- coding: utf-8 -*-
"""Generate the personalized CF-guided candidate domain C_u with the FROZEN backbone.

Major steps:
  1. Load the frozen LightCCF backbone and the per-user critique set.
  2. For each critique user, score all items via the backbone, drop the user history
     (train positives, and optionally test positives) and keep the top-N as C_u.
  3. Map each candidate item id to its title and write prepare_for_LLM/top_recommendations.json
     = {str(uid): ["id::title", ...]}, the constrained pool the LLM is allowed to choose from.
"""
import os
import sys
import json
import random
import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import utility.parser as parser
import utility.data_loader as data_loader
from LightCCF import Trainer

DATA_DIR = os.path.join(PROJ, "dataset/amazon-book-82p")
# path to the original dataset file processed_Amazonbooks.dat (NOT included in this repo)
TITLE_FILE = os.environ.get("TITLE_FILE", "processed_Amazonbooks.dat")
# critique user source / output / pool size are env-overridable (alongside TOP_N)
CRITIQUE_JSON = os.environ.get(
    "CRITIQUE_JSON", os.path.join(PROJ, "prepare_for_LLM/user_test_critique_with_keyphrase.json"))
OUT = os.environ.get("OUT", os.path.join(PROJ, "prepare_for_LLM/top_recommendations.json"))
TOP_N = int(os.environ.get("TOP_N", "200"))
# KEEP_TEST=1 -> candidate pool excludes only train history, so test positives may be ranked in.
# default 0 = legacy behavior (exclude train ∪ test).
KEEP_TEST = os.environ.get("KEEP_TEST", "0") == "1"


def load_title_map(path):
    id2title = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].isdigit():
                id2title[int(parts[0])] = parts[2]
    return id2title


def main():
    seed = 2025
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    args = parser.parse_args()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | TOP_N={TOP_N} | KEEP_TEST={KEEP_TEST}")

    dataset = data_loader.Data(args)
    print(f"  num_users={dataset.num_users}, num_items={dataset.num_items}")
    lightccf = Trainer(args, dataset, device, None)
    mp = os.environ.get("BACKBONE", "model_save/best_model_LightCCF.pth")
    print(f"Loading {mp} ...")
    lightccf.model.load_state_dict(torch.load(mp, map_location=device))
    lightccf.model.to(device).eval()
    model = lightccf.model

    id2title = load_title_map(TITLE_FILE)
    print(f"title map: {len(id2title)} entries")
    print(f"critique source: {CRITIQUE_JSON} | output: {OUT}")
    critique = json.load(open(CRITIQUE_JSON, encoding="utf-8"))
    users = sorted(int(u) for u in critique.keys() if int(u) < dataset.num_users)
    print(f"critique users: {len(users)}")

    result = {}
    empty = 0
    test_in_pool_total = 0
    users_with_test_in_pool = 0
    for i, uid in enumerate(users):
        hist = set(dataset.all_positive[uid])
        test_pos = set(dataset.test_dict.get(uid, []))
        if not KEEP_TEST:
            hist = hist | test_pos   # legacy: also exclude test
        with torch.no_grad():
            pred = model.get_rating_for_test(torch.tensor([uid], device=device)).squeeze(0).cpu().numpy()
        order = np.argsort(-pred)
        top = [int(it) for it in order if int(it) not in hist][:TOP_N]
        if KEEP_TEST:
            in_pool = [it for it in top if it in test_pos]
            if in_pool:
                users_with_test_in_pool += 1
                test_in_pool_total += len(in_pool)
        result[str(uid)] = [f"{it}::{id2title.get(it, '')}" for it in top]
        if not top:
            empty += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(users):
            print(f"  progress {i+1}/{len(users)} users")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    lens = [len(v) for v in result.values()]
    all_ids = [int(s.split("::")[0]) for v in result.values() for s in v if "::" in s]
    titled = sum(1 for v in result.values() for s in v if s.split("::", 1)[1].strip())
    total = sum(lens)
    print(f"\n=== done === {len(result)} users, per-user count min/med/max={min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    print(f"  empty-pool users: {empty} | id range [{min(all_ids)},{max(all_ids)}] (should be within [0,{dataset.num_items-1}])")
    print(f"  title coverage: {titled}/{total} ({titled/max(total,1)*100:.1f}%)")
    if KEEP_TEST:
        print(f"  * test positives in pool: {test_in_pool_total}, covering {users_with_test_in_pool}/{len(users)} users "
              f"(mean {test_in_pool_total/max(users_with_test_in_pool,1):.2f}/user)")
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
