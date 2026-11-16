"""LLM backends with stand-in HTTP responses (no network, no key): log-prob parsing, usage, cache, error handling, raw features."""
import numpy as np
import pytest

from scripts import eval_selective as ES
from src.pipeline import llm


class Resp:
    def __init__(self, payload, status=200, text=""):
        self._p, self.status_code, self.text = payload, status, text or str(payload)

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def toks(spec):
    """spec: [(token, logprob, runner_up_logprob)] -> the list both Ollama and OpenAI-style APIs return."""
    return [{"token": t, "logprob": lp, "top_logprobs": [{"token": t, "logprob": lp}, {"token": "x", "logprob": ru}]} for t, lp, ru in spec]


SPEC = [("The", -0.05, -3.0), (" answer", -0.20, -2.0), (" is", -0.01, -5.0), (" Paris", -0.40, -1.5), (".", -0.02, -6.0),
        (" It", -1.20, -1.5), (" is", -0.30, -2.0), (" big", -2.00, -2.4), (".", -0.10, -4.0)]


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_CACHE_PATH", tmp_path / "cache.sqlite")
    monkeypatch.setattr(llm, "_cache", None)


def test_ollama_logprobs_usage_and_cache(fresh_cache, monkeypatch):
    calls = []

    def post(url, json=None, **kw):
        calls.append((url, json))
        return Resp({"message": {"content": "The answer is Paris. It is big."}, "logprobs": toks(SPEC), "prompt_eval_count": 120, "eval_count": 9})
    monkeypatch.setattr(llm.requests, "post", post)
    m = llm.LLM(backend="ollama", model="qwen2.5:1.5b", logprobs=True)
    g = m.generate("Q?")
    assert g["text"] == "The answer is Paris. It is big." and g["cached"] is False
    assert calls[0][1]["logprobs"] is True and calls[0][1]["top_logprobs"] == 2
    assert g["lp"]["n"] == 9 and np.isclose(g["lp"]["mean"], np.mean([s[1] for s in SPEC]))
    assert g["lp_raw"]["t"][:3] == ["The", " answer", " is"] and len(g["lp_raw"]["lp"]) == 9 and np.isclose(g["lp_raw"]["m"][0], 2.95)
    assert g["usage"] == {"prompt_tokens": 120, "completion_tokens": 9}
    assert m.generate("Q?")["cached"] is True and len(calls) == 1             # second call comes from the cache


def test_openai_compatible_logprobs_usage_and_request_body(fresh_cache, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test-abc123")
    seen = {}

    def post(url, json=None, headers=None, **kw):
        seen.update(url=url, body=json, headers=headers)
        return Resp({"choices": [{"message": {"content": "The answer is Paris."}, "logprobs": {"content": toks(SPEC[:5])}}],
                     "usage": {"prompt_tokens": 1500, "completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 30}}})
    monkeypatch.setattr(llm.requests, "post", post)
    m = llm.LLM(backend="openai_compat", model="meta-llama/x", base_url="https://api.example.com/v1", api_key_env="TEST_API_KEY",
                logprobs=True, reasoning_effort="low", extra_body={"reasoning": {"enabled": False}}, max_tokens_param="max_completion_tokens",
                max_tokens=200)
    g = m.generate("Q?")
    b = seen["body"]
    assert seen["url"].endswith("/chat/completions") and seen["headers"]["Authorization"] == "Bearer sk-test-abc123"
    assert b["logprobs"] is True and b["top_logprobs"] == 2 and b["max_completion_tokens"] == 200 and "max_tokens" not in b
    assert b["reasoning_effort"] == "low" and b["reasoning"] == {"enabled": False} and b["temperature"] == 0 and b["seed"] == 0
    assert g["text"] == "The answer is Paris." and g["lp"]["n"] == 5 and len(g["lp_raw"]["lp"]) == 5
    assert g["usage"] == {"prompt_tokens": 1500, "completion_tokens": 40, "reasoning_tokens": 30}


def test_without_logprobs_nothing_extra_is_requested_or_stored(fresh_cache, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "k")
    seen = {}
    monkeypatch.setattr(llm.requests, "post", lambda url, json=None, **kw: seen.update(b=json) or Resp({"choices": [{"message": {"content": "ok"}}]}))
    g = llm.LLM(backend="openai_compat", model="m", base_url="https://h/v1", api_key_env="TEST_API_KEY", temperature=None, send_seed=False).generate("Q")
    assert "logprobs" not in seen["b"] and "temperature" not in seen["b"] and "seed" not in seen["b"]
    assert g["text"] == "ok" and "lp" not in g and "lp_raw" not in g and "usage" not in g


def test_http_400_is_not_retried_and_the_key_never_appears_in_the_error(fresh_cache, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-secret-999")
    n = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: n.append(1) or Resp({}, status=400, text="bad request sk-secret-999 echoed"))
    with pytest.raises(llm._Redacted) as e:
        llm.LLM(backend="openai_compat", model="m", base_url="https://h/v1", api_key_env="TEST_API_KEY").generate("Q")
    assert len(n) == 1 and "sk-secret-999" not in str(e.value)


def test_cache_key_separates_logprob_and_reasoning_setups(fresh_cache, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "k")
    mk = lambda **kw: llm.LLM(backend="openai_compat", model="m", base_url="https://h/v1", api_key_env="TEST_API_KEY", **kw)._key(None, "P")
    assert len({mk(), mk(logprobs=True), mk(reasoning_effort="low"), mk(extra_body={"a": 1})}) == 4
    assert mk(extra_body={"a": 1, "b": 2}) == mk(extra_body={"b": 2, "a": 1})


def test_raw_features_first_token_and_first_sentence():
    raw = llm._raw_logprobs(toks(SPEC))
    f = ES.raw_lp_features(raw)
    lp = [s[1] for s in SPEC]
    assert f["lp_tok1"] == -0.05
    assert np.isclose(f["lp_first_sent"], np.mean(lp[:5]))          # "The answer is Paris." = the first five tokens
    assert np.isclose(f["lp_first5"], np.mean(lp[:5]))
    assert ES.raw_lp_features(None) == {} and ES.raw_lp_features({"lp": []}) == {}
    # decimals do not end the sentence: "3.5 million." must keep "3", ".", "5"
    d = llm._raw_logprobs(toks([("It", -0.1, -2), (" is", -0.1, -2), (" 3", -0.5, -1), (".", -0.5, -1), ("5", -0.5, -1), (" million", -0.2, -1), (".", -0.1, -1), (" Next", -3, -3.5)]))
    assert np.isclose(ES.raw_lp_features(d)["lp_first_sent"], np.mean([-0.1, -0.1, -0.5, -0.5, -0.5, -0.2, -0.1]))


def test_raw_features_flow_into_the_evaluation_frame(tmp_path):
    import json
    run = tmp_path / "r1"; run.mkdir()
    rows = []
    for i in range(30):
        rows.append({"qid": f"q{i}", "domain": "squad2", "answerable": True, "answer": "A.", "refused": False, "correct": float(i % 2),
                     "retrieved": [{"chunk_id": "c0", "score": 0.5}], "gold_chunk_ids": ["c0"], "attribution": {}, "features": {"ret_top1": 0.5, "ret_mean": 0.5, "ret_gap": 0.0},
                     "lp_raw": llm._raw_logprobs(toks(SPEC[:5]))})
    (run / "squad2__vanilla_rag.scored.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")
    df = ES.load_frame("r1", "vanilla_rag", runs_dir=tmp_path)
    assert {"lp_tok1", "lp_first5", "lp_first_sent"} <= set(df.columns) and df.lp_tok1.eq(-0.05).all()
