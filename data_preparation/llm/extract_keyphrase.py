# -*- coding: utf-8 -*-
"""LLM job 1: extract critique keyphrases from each anchor's review (dual like/dislike prompts).

For each anchor in prepare_for_LLM/user_test_critique.json (carrying type, title, overall,
reviewText) the LLM extracts 1-3 keyphrases capturing WHY the reader liked/disliked the book.
Dislike anchors use the dislike prompt; like anchors use the like prompt. The result is written
to prepare_for_LLM/user_test_critique_with_keyphrase.json (each record gains "keyphrases").

Major steps:
  1. Load the critique json and any previous output (checkpoint resume, keyed by (user, anchor)).
  2. Build the (user, anchor) tasks still missing, run them on a thread pool with retries.
  3. Parse the strict-JSON model reply into a list of keyphrases and persist periodically.
This script calls an LLM API over stdlib urllib only.
"""
import json
import os
import re
import sys
import argparse
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# This script calls an LLM API. Use any OpenAI-compatible chat completions service.
# Configure the model and endpoint via environment variables LLM_MODEL and LLM_URL.
LLM_URL = os.environ.get("LLM_URL", "http://your-llm-endpoint/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "llm-model")

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN = os.path.join(PROJ, "prepare_for_LLM/user_test_critique.json")
OUT = os.path.join(PROJ, "prepare_for_LLM/user_test_critique_with_keyphrase.json")

PROMPT_DISLIKE = """You are a book review analyst. A reader gave the following book a LOW rating because they disliked it.
Extract the 1-3 MOST essential keyphrases that capture WHY the reader disliked this book.
Let the LLM judge the count: output only as many as the review clearly supports — 1 if there is one dominant reason, up to 3 only if there are clearly distinct reasons. Fewer and more precise is better; do NOT pad to a fixed number. Focus on concrete criticism (e.g. "slow pacing", "unlikeable characters", "poor writing", "pretentious tone"). Do NOT include praise.

Book title: {title}
Reader rating: {overall}/5
Review: {review}

Output a strict JSON object and nothing else: {{"dislike_reasons": ["...", "..."]}}"""

PROMPT_LIKE = """You are a book review analyst. A reader gave the following book a HIGH rating because they liked it.
Extract the 1-3 MOST essential keyphrases that capture WHY the reader liked this book.
Let the LLM judge the count: output only as many as the review clearly supports — 1 if there is one dominant reason, up to 3 only if there are clearly distinct reasons. Fewer and more precise is better; do NOT pad to a fixed number. Focus on concrete praise (e.g. "fast pacing", "compelling characters", "beautiful prose", "immersive world"). Do NOT include criticism.

Book title: {title}
Reader rating: {overall}/5
Review: {review}

Output a strict JSON object and nothing else: {{"like_reasons": ["...", "..."]}}"""


def parse_reasons(content):
    if not content:
        return None
    content = content.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    if m:
        content = m.group(1)
    else:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            content = m.group(0)
    try:
        obj = json.loads(content)
        reasons = (obj.get("dislike_reasons") or obj.get("like_reasons")
                   or obj.get("reasons") or [])
        reasons = [str(x).strip() for x in reasons if str(x).strip()]
        return reasons if reasons else None
    except Exception:
        return None


def call_llm(title, overall, review, typ, max_retries=3):
    review = (review or "")[:1500]
    tmpl = PROMPT_DISLIKE if typ == "dislike" else PROMPT_LIKE
    prompt = tmpl.format(title=title or "Unknown", overall=overall, review=review)
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 256,
    }).encode()
    for attempt in range(max_retries):
        try:
            # LLM API call
            req = urllib.request.Request(LLM_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=200) as resp:
                out = json.loads(resp.read())
            content = out["choices"][0]["message"]["content"]
            reasons = parse_reasons(content)
            if reasons:
                return reasons
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return None
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default=IN, help="input critique json (each anchor has type/title/reviewText)")
    ap.add_argument("--out", default=OUT, help="output json (each anchor gains keyphrases)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    in_path, out_path = args.in_file, args.out
    data = json.load(open(in_path, encoding="utf-8"))
    user_keys = list(data.keys())
    if args.limit:
        user_keys = user_keys[:args.limit]

    # checkpoint resume
    done = {}  # (user, anchor_idx) -> keyphrases
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path, encoding="utf-8"))
            for u, recs in prev.items():
                for i, r in enumerate(recs):
                    if r.get("keyphrases"):
                        done[(u, i)] = r["keyphrases"]
            print(f"checkpoint resume: {len(done)} anchors already have keyphrases")
        except Exception:
            pass

    tasks = []
    for u in user_keys:
        for i, r in enumerate(data[u]):
            if (u, i) not in done:
                tasks.append((u, i, r))
    print(f"to extract: {len(tasks)} anchors (workers={args.workers})")

    out_data = {u: list(recs) for u, recs in data.items() if u in user_keys}
    for (u, i), kp in done.items():
        if u in out_data and i < len(out_data[u]):
            out_data[u][i]["keyphrases"] = kp

    ok = 0; fail = 0; t0 = time.time(); last_save = 0

    def work(task):
        u, i, r = task
        kp = call_llm(r.get("title", ""), r.get("overall"), r.get("reviewText", ""), r.get("type", "dislike"))
        return (u, i, kp)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, t): t for t in tasks}
        for n, fut in enumerate(as_completed(futures), 1):
            u, i, kp = fut.result()
            if kp:
                out_data[u][i]["keyphrases"] = kp
                ok += 1
            else:
                out_data[u][i]["keyphrases"] = []
                fail += 1
            if n % 50 == 0 or n == len(tasks):
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed else 0
                eta = (len(tasks) - n) / rate if rate else 0
                print(f"  progress {n}/{len(tasks)} ok={ok} fail={fail} "
                      f"({rate:.1f}/s, ETA {eta/60:.1f}min)")
                if n - last_save >= 100:
                    json.dump(out_data, open(out_path, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                    last_save = n

    json.dump(out_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    alln = [len(r.get("keyphrases", [])) for recs in out_data.values() for r in recs]
    print(f"\n=== done === ok={ok} fail={fail}")
    if alln:
        print(f"keyphrases per anchor: mean={sum(alln)/len(alln):.1f} min={min(alln)} max={max(alln)}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
