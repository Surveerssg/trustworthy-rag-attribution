"""Shared paths, IO helpers and run bookkeeping."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
INDICES = DATA / "indices"
RUNS = ROOT / "experiments" / "runs"
RESULTS = ROOT / "experiments" / "results"
PAPER = ROOT / "paper"

DOMAINS = ["hotpotqa", "pubmedqa", "cuad", "financebench", "squad2"]


def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path):
    with open(path, encoding="utf8") as f:
        return yaml.safe_load(f)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def make_run_dir(name, config):
    """experiments/runs/<name>/ with the config and metadata frozen next to the outputs."""
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "config.yaml", "w", encoding="utf8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    meta = {"name": name, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit()}
    with open(d / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    return d


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text, min_chars=3):
    """Regex sentence splitter (no model download). Good enough for LLM answers and passages."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= min_chars]
