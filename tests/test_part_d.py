"""Part D (one large model over both T4s): configs, the nested question subsample, the time budget, the dry run and the notebook."""
import json
import sys

import pytest
import yaml

from scripts import ingest_kaggle as IK
from scripts import run_generation as RG
from src import common
from src.common import ROOT, load_config

CONFIGS = ["kaggle_v2_qwen32b", "kaggle_v2_gemma27b"]
SIX = {"vanilla_rag", "trust_rag", "removed_trust", "oracle_vanilla", "improved_vanilla", "removed_vanilla"}


@pytest.mark.parametrize("name", CONFIGS)
def test_part_d_configs(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg["name"] == name.replace("kaggle_", "")                 # the run folder the analysis scripts expect: v2_<model>
    assert cfg["llm"]["backend"] == "ollama" and cfg["llm"]["logprobs"] is True
    assert cfg["llm"]["base_url"].endswith(":11434")                    # ONE server, both GPUs
    assert cfg["use_plans"] is True and set(cfg["systems"]) == SIX and cfg["systems"][0] == "vanilla_rag"
    assert set(cfg["domains"]) == set(common.DOMAINS) and cfg["workers"] == 3


def test_subsample_is_nested_and_keeps_the_balance():
    for d in ("squad2", "cuad", "financebench"):
        small = RG.select_questions(d, {"seed": 0, "limit": 50})
        big = RG.select_questions(d, {"seed": 0, "limit": 100})
        assert len(small) == 50 and len(big) == 100
        assert {q["qid"] for q in small} <= {q["qid"] for q in big}      # a later, larger run only adds questions
    sq = RG.select_questions("squad2", {"seed": 0, "limit": 100})
    assert sum(not q["answerable"] for q in sq) == 50                    # SQuAD 2.0 is half unanswerable


# ------------------------------------------------------------------ run_generation: --dry-run and --max-hours
RAW = {"t": ["Paris"], "lp": [-0.1], "m": [2.0]}


class Clock:
    now = 1000.0

    def time(self):
        return Clock.now


class Fake:
    def __init__(self, tick=0.0):
        self.calls, self.tick = 0, tick

    def generate(self, prompt, system=None):
        self.calls += 1
        Clock.now += self.tick
        return {"text": "Paris.", "seconds": 0.01, "cached": False, "lp": {"n": 1, "mean": -0.1}, "lp_raw": RAW, "usage": None}


def invoke(tmp_path, monkeypatch, fake, *extra):
    cfg = {"name": "_t", "domains": ["squad2"], "systems": ["vanilla_rag", "removed_vanilla"], "use_plans": True, "limit": 8, "seed": 0,
           "k": 5, "llm": {"backend": "ollama", "model": "stub", "logprobs": True}, "workers": 1}
    f = tmp_path / "cfg.yaml"
    f.write_text(yaml.safe_dump(cfg), encoding="utf8")
    monkeypatch.setattr(common, "RUNS", tmp_path)
    monkeypatch.setattr(RG, "from_config", lambda c: fake)
    monkeypatch.setattr(sys, "argv", ["run_generation", "--config", str(f), *extra])
    RG.main()


def test_dry_run_counts_and_generates_nothing(tmp_path, monkeypatch, capsys):
    fake = Fake()
    invoke(tmp_path, monkeypatch, fake, "--dry-run")
    out = capsys.readouterr().out
    assert fake.calls == 0 and not (tmp_path / "_t").exists()           # no run folder, no model call
    total = int(out.split("TOTAL_TODO")[1].split()[0])
    assert total > 8 and "squad2/vanilla_rag: 8 to generate" in out


def test_limit_flag_overrides_the_config(tmp_path, monkeypatch):
    fake = Fake()
    invoke(tmp_path, monkeypatch, fake, "--limit", "4", "--systems", "vanilla_rag")
    assert len(common.read_jsonl(tmp_path / "_t" / "squad2__vanilla_rag.jsonl")) == 4
    assert yaml.safe_load((tmp_path / "_t" / "config.yaml").read_text(encoding="utf8"))["limit"] == 4   # the run records what it used


def test_time_budget_stops_cleanly_and_a_rerun_finishes(tmp_path, monkeypatch, capsys):
    Clock.now = 1000.0
    monkeypatch.setattr(RG, "time", Clock())                            # a fake clock: every answer "takes" 10 s
    fake = Fake(tick=10.0)
    invoke(tmp_path, monkeypatch, fake, "--max-hours", "0.0125", "--systems", "vanilla_rag")   # 45 s budget
    rows = common.read_jsonl(tmp_path / "_t" / "squad2__vanilla_rag.jsonl")   # every line parses: no half-written record
    assert 0 < len(rows) < 8 and "[deadline]" in capsys.readouterr().out
    invoke(tmp_path, monkeypatch, fake, "--systems", "vanilla_rag")     # no budget: continues where it stopped
    rows = common.read_jsonl(tmp_path / "_t" / "squad2__vanilla_rag.jsonl")
    assert len(rows) == 8 and len({r["qid"] for r in rows}) == 8


# ------------------------------------------------------------------ ingest completeness check with a subsampled run
def test_ingest_expected_counts_follow_the_run_config():
    cfg = {"seed": 0, "limit": 100}
    assert IK.expected_count(cfg, "squad2", "vanilla_rag") == 100
    assert IK.expected_count(cfg, "squad2", "removed_trust") == 50       # answerable half only
    assert IK.expected_count({"seed": 0, "limit": None}, "cuad", "vanilla_rag") == 300
    assert IK.expected_count(None, "cuad", "vanilla_rag") == 300 and IK.expected_count(None, "cuad", "removed_trust") is None


# ------------------------------------------------------------------ the notebook
def test_notebook_has_part_d_and_every_cell_compiles():
    nb = ROOT / "kaggle" / "trag_kaggle_stage2.ipynb"
    if not nb.exists():
        pytest.skip("notebook is not in the Kaggle bundle")
    cells = json.loads(nb.read_text(encoding="utf8"))["cells"]
    src = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in cells)
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]) if isinstance(c["source"], list) else c["source"], f"cell{i}", "exec")
    assert 'PART == "D"' in src and "D_MODEL" in src and "--max-hours" in src and "OLLAMA_SCHED_SPREAD" in src and "D_NO_LOGPROBS" in src
    assert 'PART == "S"' not in src                                       # the score-only part went with the paid-API plan
    for m in ("qwen32b", "gemma27b"):
        assert (ROOT / "configs" / f"kaggle_v2_{m}.yaml").exists()
