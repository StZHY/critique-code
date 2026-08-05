"""LLM job 2: critique-conditioned candidate-evidence generation over the constrained pool; writes batch_llm_cri_suggestions.json."""
import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

LLM_URL = os.environ.get("LLM_URL", "http://your-llm-endpoint/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "llm-model")

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJ)

WORKERS = 16
MAX_RETRIES = 3
RETRY_DELAY = 2
TEMPERATURE = 0.7
MAX_TOKENS = 2048
SUGGEST_AT_LEAST = int(os.environ.get("SUGGEST_N", "20"))
PROGRESS_EVERY = 200

POOL_FILE = os.path.join(PROJ, "prepare_for_LLM/top_recommendations.json")
CRITIQUE_FILE = os.path.join(PROJ, "prepare_for_LLM/user_test_critique_with_keyphrase.json")
OUTPUT_FILE = os.path.join(PROJ, "prepare_for_LLM/batch_llm_cri_suggestions.json")


def build_system_prompt(book_list_str, typ):
    polarity = "dislike" if typ == "dislike" else "like"
    return f"""You are a book critic and an expert on a large Amazon book catalog.
Your task is to suggest books a user might {polarity} based on their taste.
You MUST ONLY suggest books from the following list. The format for each book in the list is 'book_id::book_title'.
<BookList>
{book_list_str.strip()}
</BookList>
You MUST output your result as a strict JSON object. The key of this JSON object MUST be the ID of the {polarity}d book
provided in the user's message. The value must be a list of strings, where each string is
in the format 'book_id::book_title::score'.
For example, if the {polarity}d book ID from the user prompt is '<{polarity}_book_id>', your entire output must be a
JSON object that looks like this:
{{"<{polarity}_book_id>": ["<book_id_1>::<book_title_1>::<score_1>", "<book_id_2>::<book_title_2>::<score_2>"]}}

The 'score' should be a probability from 0.0 to 1.0 representing how likely the user is to {polarity} the book.
Do not include any other text, explanations, or formatting."""


def build_user_prompt(book_id, title, keyphrases, contrast_items, typ):
    kps = ", ".join(keyphrases) if keyphrases else "unknown"
    c0 = contrast_items[0] if len(contrast_items) > 0 else "N/A"
    c1 = contrast_items[1] if len(contrast_items) > 1 else "N/A"
    if typ == "dislike":
        return (
            f"The ID of the book that the user DISLIKED is:\n"
            f"<DislikedBookID>\n{book_id}\n</DislikedBookID>\n"
            f"The title of the disliked book is:\n<DislikedBookTitle>\n{title}\n</DislikedBookTitle>\n"
            f"The critique reasons (why the user disliked it) are: '{kps}'.\n"
            f"The two books the user LIKED for comparison are: '{c0}' and '{c1}'.\n"
            f"Based on this, your task is to suggest at least {SUGGEST_AT_LEAST} books from the list that the user might also DISLIKE.\n"
            f"These suggestions should be thematically or stylistically consistent with the disliked book and its critique reasons,\n"
            f"and in stark contrast to the positive comparison books."
        )
    else:
        return (
            f"The ID of the book that the user LIKED is:\n"
            f"<LikedBookID>\n{book_id}\n</LikedBookID>\n"
            f"The title of the liked book is:\n<LikedBookTitle>\n{title}\n</LikedBookTitle>\n"
            f"The reasons why the user liked it are: '{kps}'.\n"
            f"The two books the user DISLIKED for comparison are: '{c0}' and '{c1}'.\n"
            f"Based on this, your task is to suggest at least {SUGGEST_AT_LEAST} books from the list that the user might also LIKE.\n"
            f"These suggestions should be thematically or stylistically consistent with the liked book and its like reasons,\n"
            f"and in stark contrast to the negative comparison books (i.e. avoid traits of the disliked books)."
        )


def parse_json(content):
    """Extract JSON from the model reply, tolerating ```json``` fences and bare JSON; None on failure."""
    if not content:
        return None
    content = content.strip()
    start = content.find('```json')
    end = content.rfind('```')
    if start != -1 and end != -1 and end > start:
        content = content[start + len('```json'):end].strip()
    else:
        s = content.find('```')
        e = content.rfind('```')
        if s != -1 and e != -1 and e > s:
            content = content[s + 3:e].strip()
    first = content.find('{')
    last = content.rfind('}')
    if first != -1 and last != -1 and last > first:
        content = content[first:last + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def call_llm(messages, max_retries=MAX_RETRIES):
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }).encode()
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(LLM_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read())
            content = out["choices"][0]["message"]["content"]
            data = parse_json(content)
            if data is not None:
                return data
            last_err = "JSON parse failed"
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return None
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(RETRY_DELAY * (attempt + 1))
    print(f"  call_llm failed: {last_err}", file=sys.stderr)
    return None


def build_candidate_list(pool, anchor_book_id):
    """Drop the anchor book from this user's top-N pool (by id prefix, so the model cannot echo it)."""
    exclude = str(anchor_book_id)
    kept = [e for e in pool if e.split('::', 1)[0] != exclude]
    return "\n".join(kept)


def main():
    ap = argparse.ArgumentParser(description="constrained-pool LLM candidate generation (positive/negative symmetric)")
    ap.add_argument('--pool_file', default=POOL_FILE)
    ap.add_argument('--critique_file', default=CRITIQUE_FILE)
    ap.add_argument('--output_file', default=OUTPUT_FILE)
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"read candidate pool: {args.pool_file}")
    pool_data = json.load(open(args.pool_file, encoding='utf-8'))
    print(f"read critique: {args.critique_file}")
    critique_data = json.load(open(args.critique_file, encoding='utf-8'))

    user_keys = list(critique_data.keys())
    if args.limit:
        user_keys = user_keys[:args.limit]

    out_data = {u: {} for u in user_keys}
    done = set()
    if os.path.exists(args.output_file):
        try:
            prev = json.load(open(args.output_file, encoding='utf-8'))
            for u in user_keys:
                if u in prev:
                    out_data[u] = dict(prev[u])
                    for bid in prev[u]:
                        done.add((u, str(bid)))
            print(f"checkpoint resume: {len(done)} anchors already have suggestions")
        except Exception:
            pass

    tasks = []
    skipped = 0
    no_pool = 0
    for u in user_keys:
        pool = pool_data.get(u, [])
        if not pool:
            no_pool += 1
            continue
        for info in critique_data.get(u, []):
            book_id = info.get("book_id")
            title = info.get("title", "")
            typ = info.get("type", "dislike")
            keyphrases = info.get("keyphrases", [])
            contrast_items = info.get("contrast_items", [])
            if book_id is None or len(contrast_items) < 2:
                skipped += 1
                continue
            if (u, str(book_id)) in done:
                skipped += 1
                continue
            cand = build_candidate_list(pool, book_id)
            if not cand.strip():
                no_pool += 1
                continue
            messages = [
                {"role": "system", "content": build_system_prompt(cand, typ)},
                {"role": "user", "content": build_user_prompt(
                    book_id, title, keyphrases, contrast_items, typ)},
            ]
            tasks.append((u, book_id, messages))

    print(f"to generate: {len(tasks)} anchors (workers={args.workers}); skipped {skipped}, no pool {no_pool}")
    if not tasks:
        print("no pending tasks, exiting.")
        json.dump(out_data, open(args.output_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return

    ok = 0; fail = 0; total_sug = 0; last_save = 0

    def work(task):
        u, book_id, messages = task
        data = call_llm(messages)
        return (u, book_id, data)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, t): t for t in tasks}
        for n, fut in enumerate(as_completed(futures), 1):
            u, book_id, data = fut.result()
            if isinstance(data, dict):
                out_data[u].update({str(k): v for k, v in data.items()})
                ok += 1
                for v in data.values():
                    if isinstance(v, list):
                        total_sug += len(v)
            else:
                out_data[u][str(book_id)] = None
                fail += 1
            if n % PROGRESS_EVERY == 0 or n == len(tasks):
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed else 0
                eta = (len(tasks) - n) / rate if rate else 0
                print(f"  progress {n}/{len(tasks)} ok={ok} fail={fail} "
                      f"avg_sug={total_sug / max(ok, 1):.1f} ({rate:.1f}/s, ETA {eta / 60:.1f}min)")
                if n - last_save >= 200:
                    json.dump(out_data, open(args.output_file, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                    last_save = n

    json.dump(out_data, open(args.output_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    elapsed = time.time() - t0
    print(f"\n=== done === ok={ok} fail={fail} total suggested books={total_sug} "
          f"avg/anchor={total_sug / max(ok, 1):.1f} elapsed {elapsed:.0f}s")
    print(f"output: {args.output_file}")


if __name__ == "__main__":
    main()
