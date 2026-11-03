"""run_generation.py end to end with a stand-in model: new log-prob/usage fields reach the output, removal runs only on answerable
questions, and a second run resumes instead of regenerating. Writes to a temp folder only."""
import sys

import yaml

from scripts import run_generation as RG
from src import common

RAW = {"t": ["Paris", "."], "lp": [-0.1, -0.2], "m": [2.0, 3.0]}


class Fake:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system=None):
        self.calls += 1
        return {"text": "Paris.", "seconds": 0.01, "cached": False, "lp": {"n": 2, "mean": -0.15}, "lp_raw": RAW,
                "usage": {"prompt_tokens": 900, "completion_tokens": 3}}


def run(tmp_path, monkeypatch, fake):
    cfg = {"name": "_t", "domains": ["squad2"], "systems": ["vanilla_rag", "removed_vanilla"], "use_plans": True, "limit": 8, "seed": 0,
           "k": 5, "llm": {"backend": "openai_compat", "model": "stub", "logprobs": True}, "workers": 1}
    f = tmp_path / "cfg.yaml"
    f.write_text(yaml.safe_dump(cfg), encoding="utf8")
    monkeypatch.setattr(common, "RUNS", tmp_path)
    monkeypatch.setattr(RG, "from_config", lambda c: fake)
    monkeypatch.setattr(sys, "argv", ["run_generation", "--config", str(f)])
    RG.main()


def test_new_fields_reach_the_output_and_runs_resume(tmp_path, monkeypatch):
    fake = Fake()
    run(tmp_path, monkeypatch, fake)
    van = common.read_jsonl(tmp_path / "_t" / "squad2__vanilla_rag.jsonl")
    rem = common.read_jsonl(tmp_path / "_t" / "squad2__removed_vanilla.jsonl")
    assert len(van) == 8 and all(r["lp_raw"] == RAW and r["usage"]["prompt_tokens"] == 900 and r["lp"]["n"] == 2 for r in van)
    assert 0 < len(rem) < 8 and all(not r["answerable"] and r["meta"]["original_gold_answers"] for r in rem)   # removal: answerable questions only
    first = fake.calls
    run(tmp_path, monkeypatch, fake)
    assert fake.calls == first                                   # nothing regenerated on the second run
