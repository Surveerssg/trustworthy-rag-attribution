"""One command after Kaggle finishes:  python -m scripts.ingest_kaggle trag_results_core.zip

1. unzips runs/results into the project (never overwrites a local file that is newer/larger)
2. checks every (domain, system) file is complete for each run
3. runs every analysis for each run and rebuilds figures
"""
import subprocess
import sys
import zipfile
from pathlib import Path

from src.common import ROOT, RUNS, load_config, read_jsonl

ANALYSES = [["scripts.analyze_crossdomain", "--system", "vanilla_rag"], ["scripts.analyze_crossdomain", "--system", "trust_rag"],
            ["scripts.analyze_refusal"], ["scripts.analyze_systems"], ["scripts.analyze_extras"],
            ["scripts.error_taxonomy", "--system", "vanilla_rag"]]
EXPECTED = {"squad2": 500, "hotpotqa": 200, "pubmedqa": 200, "cuad": 300, "financebench": 150}


def expected_count(cfg, domain, system):
    """Records a complete (domain, system) file must hold: the questions this run selected (a run may use a seeded
    subsample, `limit`); the evidence-removal systems run on the answerable ones only."""
    if cfg is None:
        return None if system.startswith("removed") else EXPECTED[domain]
    from scripts.run_generation import select_questions
    qs = select_questions(domain, cfg)
    return sum(q["answerable"] for q in qs) if system.startswith("removed") else len(qs)


def main(zpath, analyses=True):
    z = zipfile.ZipFile(zpath)
    runs = set()
    for n in z.namelist():
        if n.startswith("logs/ollama_"):   # server logs: hundreds of MB of prompt-cache chatter; left inside the zip
            print(f"skip {n} ({z.getinfo(n).file_size / 1e6:.0f} MB, stays in the zip)"); continue
        if n.startswith("logs/"):
            dst = ROOT / "experiments" / "kaggle_logs" / Path(n).name
        else:
            dst = ROOT / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > z.getinfo(n).file_size:
            print("keep local (larger):", dst.relative_to(ROOT)); continue
        dst.write_bytes(z.read(n))
        if n.startswith("experiments/runs/"):
            runs.add(n.split("/")[2])
    print("runs:", sorted(runs))
    ok = True
    for run in sorted(runs):
        cfg_path = RUNS / run / "config.yaml"
        cfg = load_config(cfg_path) if cfg_path.exists() else None
        for f in sorted((RUNS / run).glob("*__*.scored.jsonl")):
            d = f.name.split("__")[0]; s = f.name.split("__")[1].replace(".scored.jsonl", "")
            n = len(read_jsonl(f)); exp = expected_count(cfg, d, s)
            flag = "" if exp is None or n == exp else f"  <-- expected {exp}"
            ok &= not flag
            print(f"  {run}/{f.name}: {n}{flag}")
    if analyses:
        for run in sorted(runs):
            for a in ANALYSES:
                print(f"\n######## {run}: {' '.join(a)}")
                subprocess.run([sys.executable, "-W", "ignore", "-m", a[0], "--run", run, *a[1:]], cwd=ROOT)
        subprocess.run([sys.executable, "-m", "scripts.make_figures"], cwd=ROOT)
    print("\nALL FILES COMPLETE" if ok else "\nSOME FILES INCOMPLETE - see above")


if __name__ == "__main__":
    # --no-analyses: unpack and check completeness only (the v2 runs lack trust_rag/closed_book, which the legacy
    # analyses expect; use scripts.analyze_stage2 for them instead)
    main(sys.argv[1], analyses="--no-analyses" not in sys.argv)
