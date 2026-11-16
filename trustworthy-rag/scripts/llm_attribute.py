"""LLM-as-attributor baseline (Part F, pre-registered in paper/findings/analysis_plan_llm_attributor.md).

python -m scripts.llm_attribute items                                    # laptop: build data/llm_attr/{items,chunks}.jsonl from the scored oracle runs
python -m scripts.llm_attribute --config configs/kaggle_attr_qwen32b.yaml [--max-hours H] [--dry-run]   # Kaggle: run the judge (Ollama)

For every answer sentence of every oracle_vanilla record that the paper's attribution analyses use (answerable, not refused, a gold chunk in the
context), the judge is asked which of the five passages supports the sentence. Writes experiments/runs/<name>/<domain>__judge.jsonl, one row per
sentence: the probability of each passage number in the first output token, the pick, the raw reply. Resumable (finished ids are skipped), stops
cleanly at the time budget, prints TOTAL_TODO with --dry-run (the Kaggle notebook reads it).
"""
import argparse
import glob
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from src.common import DATA, RUNS, append_jsonl, load_config, make_run_dir, read_jsonl, write_jsonl
from src.pipeline.prompts import format_passages

ITEMS, CHUNKS = DATA / "llm_attr" / "items.jsonl", DATA / "llm_attr" / "chunks.jsonl"
PROMPT = ("Below are numbered passages and one sentence taken from an answer. Which passage supports the sentence?\n"
          "Reply with the passage number only.\n\n{passages}\n\nSentence: {sentence}\nPassage number:")


def build_items():
    """Laptop step: the exact sentences and contexts already scored, so every method is compared on the same sentences."""
    from src.attribution.fusion import acc_by_weight
    from src.attribution.methods import refresh_record
    items, chunks = [], {}
    files = sorted(glob.glob(str(RUNS / "v2_*" / "*__oracle_vanilla.scored.jsonl")) + glob.glob(str(RUNS / "tq_*" / "*__oracle_vanilla.scored.jsonl")))
    for f in files:
        run, domain = os.path.basename(os.path.dirname(f)), os.path.basename(f).split("__")[0]
        for r in read_jsonl(f):
            if not r["answerable"] or r["refused"] or acc_by_weight(refresh_record(r)) is None:
                continue
            for c in r["retrieved"]:
                chunks[c["chunk_id"]] = {"chunk_id": c["chunk_id"], "title": c.get("title", ""), "text": c["text"]}
            for i, s in enumerate(r["attribution"]["sentences"]):
                items.append({"id": f"{run}|{domain}|{r['qid']}|{i}", "run": run, "domain": domain, "qid": r["qid"], "i": i,
                              "sentence": s["text"], "chunk_ids": [c["chunk_id"] for c in r["retrieved"]]})
    write_jsonl(ITEMS, items)
    write_jsonl(CHUNKS, list(chunks.values()))
    print(f"{len(items)} sentences from {len({(x['run'], x['domain'], x['qid']) for x in items})} answers; {len(chunks)} distinct chunks")


def parse(reply, k):
    """Probability of each passage number in the first output token; falls back to the first digit 1..k in the text."""
    probs = [0.0] * k
    lp = reply.get("logprobs") or []
    if lp:
        first = lp[0]
        for alt in [first] + list(first.get("top_logprobs") or []):
            t = (alt.get("token") or "").strip()
            # ASCII digits only: str.isdigit() is also True for '₂', '²', '٢', ... which int() rejects (crashed Part F on 31 Oct)
            if re.fullmatch(r"[0-9]+", t) and 1 <= int(t) <= k and alt.get("logprob") is not None:
                probs[int(t) - 1] = max(probs[int(t) - 1], math.exp(alt["logprob"]))
    text = (reply.get("message") or {}).get("content", "")
    if sum(probs) > 0:
        return probs, max(range(k), key=lambda j: probs[j]) + 1, True
    m = re.search(r"[1-%d]" % k, text)
    if m:
        probs[int(m.group()) - 1] = 1.0
        return probs, int(m.group()), True
    return probs, None, False


def run_judge(cfg, max_hours=None, dry_run=False):
    import requests
    llm = cfg["llm"]
    items = read_jsonl(ITEMS)
    chunks = {c["chunk_id"]: c for c in read_jsonl(CHUNKS)}
    run = RUNS / cfg["name"] if dry_run else make_run_dir(cfg["name"], cfg)
    deadline = time.time() + max_hours * 3600 if max_hours is not None else None
    total = 0
    for domain in cfg["domains"]:
        out = run / f"{domain}__judge.jsonl"
        done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
        todo = [it for it in items if it["domain"] == domain and it["id"] not in done]
        total += len(todo)
        print(f"{domain}: {len(todo)} to judge ({len(done)} done)", flush=True)
        if dry_run or not todo:
            continue
        if deadline is not None and time.time() >= deadline:
            print("[deadline] time budget used; progress saved, rerun to continue")
            return

        def work(it):
            if deadline is not None and time.time() >= deadline:
                return None
            ctx = [chunks[c] for c in it["chunk_ids"]]
            body = {"model": llm["model"], "stream": False, "logprobs": True, "top_logprobs": 10,
                    "messages": [{"role": "user", "content": PROMPT.format(passages=format_passages(ctx), sentence=it["sentence"])}],
                    "options": {"temperature": 0, "seed": 0, "num_predict": 4, "num_ctx": llm.get("num_ctx", 4096)}}
            t0 = time.time()
            for attempt in range(6):
                try:
                    r = requests.post(f"{llm.get('base_url', 'http://localhost:11434')}/api/chat", json=body, timeout=600)
                    r.raise_for_status()
                    break
                except Exception as e:
                    if attempt == 5:
                        raise
                    print(f"[retry {attempt + 1}] {str(e)[:200]}", flush=True)
                    time.sleep(5 * (attempt + 1))
            j = r.json()
            try:
                probs, pick, ok = parse(j, len(ctx))
            except Exception as e:   # one odd reply must not stop the run: record it as a miss (counted in the format-failure rate)
                print(f"[parse error] {it['id']}: {str(e)[:120]}", flush=True)
                probs, pick, ok = [0.0] * len(ctx), None, False
            return {"id": it["id"], "pick": pick, "probs": [round(p, 5) for p in probs], "ok": ok,
                    "text": (j.get("message") or {}).get("content", "")[:40], "seconds": round(time.time() - t0, 3),
                    "prompt_tokens": j.get("prompt_eval_count")}

        t0, n = time.time(), 0
        with ThreadPoolExecutor(cfg.get("workers", 3)) as ex:
            for rec in ex.map(work, todo):
                if rec is not None:
                    append_jsonl(out, rec)
                    n += 1
                    if n % 500 == 0:
                        print(f"[{domain}] {n}/{len(todo)} in {(time.time() - t0) / 60:.0f} min", flush=True)
        print(f"[{domain}] {n} judged in {time.time() - t0:.0f}s", flush=True)
        if deadline is not None and time.time() >= deadline:
            print("[deadline] time budget used; progress saved, rerun to continue")
            return
    if dry_run:
        print(f"TOTAL_TODO {total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="judge", choices=["items", "judge"])
    ap.add_argument("--config")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.mode == "items":
        build_items()
    else:
        run_judge(load_config(a.config), a.max_hours, a.dry_run)
