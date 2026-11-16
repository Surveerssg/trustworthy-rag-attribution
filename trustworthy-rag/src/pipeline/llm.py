"""LLM backends: Ollama (local) or any OpenAI-compatible endpoint (Groq, OpenRouter, DeepInfra, Together, Fireworks, OpenAI, xAI, ...).

All calls are deterministic (temperature 0) and cached on disk keyed by (backend, model, prompt),
so re-scoring or re-running an experiment never re-pays generation cost.

Token log-probabilities (`logprobs: true` in the llm config) work on Ollama and on every OpenAI-compatible endpoint that returns
`choices[0].logprobs.content`. Each answer then carries `lp` (a small summary, used by the analyses) and `lp_raw` (every token, its
log-probability and the margin to the runner-up, so new features can be built offline without regenerating). `usage` records the
token counts the API reported, so the real cost of a run can be computed.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time

import requests

from src.common import DATA, ROOT

_CACHE_PATH = DATA / "llm_cache.sqlite"
_lock = threading.Lock()


class _Cache:
    def __init__(self):
        self.db = sqlite3.connect(_CACHE_PATH, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k):
        with _lock:
            r = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k, v):
        with _lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
            self.db.commit()


_cache = None


def _load_env():
    """Read KEY=VALUE lines from ROOT/.env into os.environ (without overriding). Values are never printed."""
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8-sig").splitlines():   # utf-8-sig: Notepad may add a BOM
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class _Redacted(Exception):
    pass


class QuotaExhausted(Exception):
    """Daily free-tier quota used up: stop cleanly; runs are resumable."""


_THOUGHT = __import__("re").compile(r"<thought>.*?(</thought>|$)", __import__("re").S)


class _RateLimiter:
    """Client-side requests-per-minute limiter shared by all worker threads."""

    def __init__(self, rpm):
        self.interval, self.next_t, self.lock = (60.0 / rpm if rpm else 0.0), 0.0, threading.Lock()

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.time()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        time.sleep(max(0.0, t - now))


class LLM:
    def __init__(self, backend="ollama", model="llama3.2:3b", max_tokens=256, num_ctx=4096,
                 base_url=None, api_key_env=None, rpm=None, logprobs=False, reasoning_effort=None, extra_body=None,
                 max_tokens_param="max_tokens", temperature=0, send_seed=True):
        self.limiter = _RateLimiter(rpm)
        self.logprobs = logprobs  # record token log-probabilities as a confidence signal
        global _cache
        _cache = _cache or _Cache()
        self.backend, self.model, self.max_tokens, self.num_ctx = backend, model, max_tokens, num_ctx
        self.reasoning_effort, self.extra_body = reasoning_effort, dict(extra_body or {})
        self.max_tokens_param, self.temperature, self.send_seed = max_tokens_param, temperature, send_seed
        self.base_url = base_url or ("http://localhost:11434" if backend == "ollama"
                                     else "https://api.groq.com/openai/v1")
        _load_env()
        self.api_key = os.environ.get(api_key_env or "GROQ_API_KEY", "")
        if backend != "ollama" and not self.api_key:
            raise RuntimeError(f"{api_key_env} not set: create .env from .env.example (never paste keys in chat)")

    def _key(self, system, prompt):
        key = [self.backend, self.model, self.max_tokens, system, prompt] + (["lp", "raw"] if self.logprobs else [])
        if self.reasoning_effort is not None or self.extra_body:   # a different thinking setup is a different experiment
            key += [self.reasoning_effort, json.dumps(self.extra_body, sort_keys=True)]
        h = hashlib.sha256(json.dumps(key).encode())
        return h.hexdigest()

    def generate(self, prompt, system=None):
        """Returns {text, seconds, cached} plus lp / lp_raw / usage when available."""
        k = self._key(system, prompt)
        hit = _cache.get(k)
        if hit is not None:
            return {**hit, "cached": True}
        t0 = time.time()
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        for attempt in range(12):
            try:
                self.limiter.wait()
                res = self._call(msgs)
                break
            except Exception as e:  # rate limits / transient server errors
                msg = str(e).replace(self.api_key, "***") if self.api_key else str(e)
                if "HTTP 400" in msg:  # malformed request: retrying cannot help
                    raise _Redacted(msg[:400]) from None
                if "per day" in msg.lower() or "perday" in msg.lower().replace(" ", ""):
                    raise QuotaExhausted(msg[:300]) from None
                if attempt == 11:
                    raise _Redacted(msg) from None
                time.sleep(min(2 ** attempt * 2, 90))
                print(f"[llm] retry {attempt + 1}: {msg[:300]}")
        out = {"text": _THOUGHT.sub("", res["text"]).strip(), "seconds": round(time.time() - t0, 3)}
        for src, dst in (("lp", "lp"), ("raw", "lp_raw"), ("usage", "usage")):
            if res.get(src) is not None:
                out[dst] = res[src]
        _cache.put(k, out)
        return {**out, "cached": False}

    def _call(self, msgs):
        """One request. Returns {"text", "lp" (summary or None), "raw" (per-token or None), "usage" (token counts or None)}."""
        if self.backend == "ollama":
            body = {"model": self.model, "messages": msgs, "stream": False,
                    "options": {"temperature": 0, "seed": 0, "num_predict": self.max_tokens, "num_ctx": self.num_ctx}}
            if self.logprobs:
                body.update(logprobs=True, top_logprobs=2)
            r = requests.post(f"{self.base_url}/api/chat", timeout=600, json=body)
            r.raise_for_status()
            j = r.json()
            toks = (j.get("logprobs") or []) if self.logprobs else []
            usage = {"prompt_tokens": j.get("prompt_eval_count"), "completion_tokens": j.get("eval_count")}
            return {"text": j["message"]["content"], "lp": _summarise_logprobs(toks) if self.logprobs else None,
                    "raw": _raw_logprobs(toks) if self.logprobs else None,
                    "usage": usage if usage["prompt_tokens"] is not None else None}
        body = {"model": self.model, "messages": msgs, self.max_tokens_param: self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.send_seed and "googleapis" not in self.base_url:  # Gemini's OpenAI-compatible endpoint rejects `seed`
            body["seed"] = 0
        if self.logprobs:
            body.update(logprobs=True, top_logprobs=2)
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        body.update(self.extra_body)   # provider-specific switches, e.g. {"reasoning": {"enabled": false}} on Together
        r = requests.post(f"{self.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=120 if not self.reasoning_effort else 600)
        if r.status_code == 429:
            if "PerDay" in r.text:  # Google puts the daily-quota id in error details, not the message
                raise RuntimeError(f"rate limited per day: {r.text[:200]}")
            raise RuntimeError(f"rate limited: {r.text[:200]}")
        if r.status_code >= 400:  # never use raise_for_status: keep request details out of tracebacks
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        choice = j["choices"][0]
        toks = ((choice.get("logprobs") or {}).get("content") or []) if self.logprobs else []
        u = j.get("usage") or {}
        usage = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                 "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")} if u else None
        return {"text": (choice.get("message") or {}).get("content") or "",
                "lp": _summarise_logprobs(toks) if toks else None, "raw": _raw_logprobs(toks) if toks else None, "usage": usage}


def _summarise_logprobs(toks):
    """Compact confidence summary of the generated tokens (kept small so the cache stays small)."""
    import math
    lps = [t["logprob"] for t in toks if t.get("logprob") is not None]
    if not lps:
        return None
    margins = []
    for t in toks:
        top = [x["logprob"] for x in (t.get("top_logprobs") or []) if x.get("logprob") is not None]
        if len(top) >= 2:
            margins.append(top[0] - top[1])
    first = lps[:12]  # the answer-bearing opening tokens
    return {"n": len(lps), "mean": sum(lps) / len(lps), "min": min(lps), "first_mean": sum(first) / len(first),
            "p_seq_first": math.exp(sum(first)), "frac_low": sum(l < math.log(0.5) for l in lps) / len(lps),
            "margin_mean": (sum(margins) / len(margins)) if margins else None}


def _raw_logprobs(toks, max_tokens=400):
    """Every generated token with its log-probability and the margin to the runner-up (None if no alternatives were returned).
    Stored so features such as the first token or the first sentence can be computed offline: {"t": [...], "lp": [...], "m": [...]}."""
    t_, lp_, m_ = [], [], []
    for t in toks[:max_tokens]:
        if t.get("logprob") is None:
            continue
        top = [x["logprob"] for x in (t.get("top_logprobs") or []) if x.get("logprob") is not None]
        t_.append(t.get("token", "")); lp_.append(round(float(t["logprob"]), 4))
        m_.append(round(top[0] - top[1], 3) if len(top) >= 2 else None)
    return {"t": t_, "lp": lp_, "m": m_} if lp_ else None


def from_config(cfg):
    c = cfg.get("llm", {})
    return LLM(backend=c.get("backend", "ollama"), model=c.get("model", "llama3.2:3b"),
               max_tokens=c.get("max_tokens", 256), num_ctx=c.get("num_ctx", 4096),
               base_url=c.get("base_url"), api_key_env=c.get("api_key_env"), rpm=c.get("rpm"),
               logprobs=c.get("logprobs", False), reasoning_effort=c.get("reasoning_effort"), extra_body=c.get("extra_body"),
               max_tokens_param=c.get("max_tokens_param", "max_tokens"), temperature=c.get("temperature", 0),
               send_seed=c.get("send_seed", True))
