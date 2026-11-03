"""Stage 1 (expensive): retrieval + generation for every (domain, system) in a config.

python -m scripts.run_generation --config configs/main.yaml [--limit 100] [--max-hours 9] [--dry-run]
Writes experiments/runs/<name>/<domain>__<system>.jsonl (resumable: finished qids are skipped).
--limit N      questions per domain: a seeded subsample (answerable/unanswerable balance kept) that is the same for every
               model and nested (a larger limit only adds questions), so a run can be extended later without redoing work
--max-hours H  stop cleanly after H hours (a Kaggle session is capped at 12 h); progress is saved, rerun to continue
--dry-run      print how many answers are still to be generated, then exit (no model needed)
"""
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:  # API-only runs on a machine without the full environment: the progress bar is optional
    def tqdm(it, **kw):
        return it

from src import common as _common
from src.common import DATA, PROC, append_jsonl, load_config, make_run_dir, read_jsonl
from src.pipeline import prompts as P
try:  # index-free environments (Colab bundle) only use precomputed plans
    from src.pipeline.index import Retriever
    from src.pipeline.retrieval_plus import PlusRetriever
except ImportError:
    Retriever = PlusRetriever = None
from src.pipeline.llm import QuotaExhausted, from_config


def select_questions(domain, cfg):
    qs = read_jsonl(PROC / domain / "questions.jsonl")
    lim = (cfg.get("limit_per_domain") or {}).get(domain, cfg.get("limit"))
    if lim and lim < len(qs):
        rng = random.Random(cfg.get("seed", 0))
        # keep answerable/unanswerable balance when subsampling
        a = [q for q in qs if q["answerable"]]; u = [q for q in qs if not q["answerable"]]
        rng.shuffle(a); rng.shuffle(u)
        na = round(lim * len(a) / len(qs))
        qs = a[:na] + u[:lim - na]
    return qs


def build_prompt(system, q, chunks):
    at = q["answer_type"]
    if system == "closed_book":
        return P.closed_book(q["question"], at)
    if system in ("vanilla_rag", "oracle_vanilla", "removed_vanilla", "improved_vanilla"):
        return P.vanilla_rag(q["question"], chunks, at)
    if system in ("trust_rag", "oracle_trust", "removed_trust", "improved_trust"):
        return P.trust_rag(q["question"], chunks, at)
    raise ValueError(system)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--systems", nargs="+", default=None, help="override the config's system list (e.g. core only)")
    ap.add_argument("--limit", type=int, default=None, help="override the config's questions per domain")
    ap.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this many hours (resumable)")
    ap.add_argument("--dry-run", action="store_true", help="only print how many answers are still to generate")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.systems:
        cfg["systems"] = args.systems
    if args.limit:
        cfg["limit"] = args.limit
    deadline = time.time() + args.max_hours * 3600 if args.max_hours is not None else None
    run = _common.RUNS / cfg["name"] if args.dry_run else make_run_dir(cfg["name"], cfg)   # config.yaml records the limit actually used
    llm = None if args.dry_run else from_config(cfg)
    k, workers = cfg.get("k", 5), cfg.get("workers", 1)

    # retrieval first (cheap), then generation with SYSTEMS as the outer loop so that, under a daily API
    # quota, the most important system is completed for every domain before the next one starts.
    plan = {}
    for domain in cfg["domains"]:
        qs = select_questions(domain, cfg)
        pf = DATA / "plans" / f"{domain}.jsonl"
        if cfg.get("use_plans") and pf.exists():  # precomputed, byte-identical contexts (see scripts/export_plan.py)
            byq = {r["qid"]: r for r in read_jsonl(pf)}
            plan[domain] = {"qs": qs, "mode": "dense", **{c: [byq[q["qid"]].get(c) for q in qs] for c in ("main", "oracle", "removed", "improved")}}
            continue
        mode = (cfg.get("retrieval_mode_per_domain") or {}).get(domain, cfg.get("retrieval_mode", "dense"))
        retr = Retriever(domain, cfg.get("chunk_words", 180)) if mode == "dense" else PlusRetriever(domain, mode)
        plan[domain] = {"qs": qs, "mode": mode, "main": retr.retrieve_many([q["question"] for q in qs], k=k)}
        if any(s.startswith("oracle") for s in cfg["systems"]):  # gold chunks forced into top-k
            plan[domain]["oracle"] = PlusRetriever(domain, "oracle").retrieve_many(
                [q["question"] for q in qs], k=k, gold=[q.get("gold_chunk_ids", []) for q in qs])
        if any(s.startswith("removed") for s in cfg["systems"]):  # evidence-removal counterfactual (answerable qs only)
            plan[domain]["removed"] = PlusRetriever(domain, "removed").retrieve_many(
                [q["question"] for q in qs], k=k, gold=[q.get("gold_chunk_ids", []) for q in qs],
                gold_docs=[q.get("gold_doc_ids", []) for q in qs])

    todo_total = 0
    for system in cfg["systems"]:
        for domain in cfg["domains"]:
            qs, mode = plan[domain]["qs"], plan[domain]["mode"]
            kind = next((p for p in ("oracle", "removed", "improved") if system.startswith(p)), "main")
            hits = plan[domain][kind]
            out = run / f"{domain}__{system}.jsonl"
            done = {r["qid"] for r in read_jsonl(out)} if out.exists() else set()
            todo = [(q, h) for q, h in zip(qs, hits) if q["qid"] not in done
                    and (kind != "removed" or q["answerable"])]
            if args.dry_run:
                todo_total += len(todo)
                print(f"{domain}/{system}: {len(todo)} to generate ({len(done)} done)")
                continue
            if not todo:
                continue
            if deadline is not None and time.time() >= deadline:
                print("[deadline] time budget used; progress saved, rerun to continue")
                return

            def work(item, system=system, domain=domain, mode=mode, kind=kind):
                if deadline is not None and time.time() >= deadline:
                    return None   # budget used: leave the rest for the next run
                q, h = item
                chunks = [] if system == "closed_book" else h
                g = llm.generate(build_prompt(system, q, chunks))
                removed = kind == "removed"
                meta = dict(q.get("meta", {}))
                if removed:  # gold evidence deleted -> should refuse; keep original gold for residual checks
                    meta["original_gold_answers"] = q["gold_answers"]
                    ctx = " ".join(c["text"] for c in h).lower()
                    meta["residual_answer_in_context"] = any(
                        len(a) > 2 and a.lower() in ctx for a in q["gold_answers"]) if q["answer_type"] != "label" else False
                return {"qid": q["qid"], "domain": domain, "system": system, "question": q["question"],
                        "answer_type": q["answer_type"], "answerable": False if removed else q["answerable"],
                        "gold_answers": [] if removed else q["gold_answers"],
                        "gold_chunk_ids": [] if removed else q.get("gold_chunk_ids", []),
                        "retrieved": h, "answer": g["text"], "refused": P.is_refusal(g["text"]),
                        "gen_seconds": g["seconds"], "cached": g["cached"], "meta": meta,
                        "retrieval_mode": kind if kind != "main" else mode, "lp": g.get("lp"),
                        "lp_raw": g.get("lp_raw"), "usage": g.get("usage")}

            t0 = time.time()
            try:
                with ThreadPoolExecutor(workers) as ex:
                    for rec in tqdm(ex.map(work, todo), total=len(todo), desc=f"{domain}/{system}"):
                        if rec is not None:
                            append_jsonl(out, rec)
            except QuotaExhausted as e:
                print(f"[quota] daily limit reached during {domain}/{system}; progress saved, rerun tomorrow. {e}")
                return
            print(f"[{domain}/{system}] {len(todo)} generated in {time.time() - t0:.0f}s")
            if deadline is not None and time.time() >= deadline:
                print("[deadline] time budget used; progress saved, rerun to continue")
                return
    if args.dry_run:
        print(f"TOTAL_TODO {todo_total}")


if __name__ == "__main__":
    main()
