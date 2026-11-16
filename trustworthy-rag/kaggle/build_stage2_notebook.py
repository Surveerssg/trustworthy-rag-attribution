"""Builds kaggle/trag_kaggle_stage2.ipynb (python kaggle/build_stage2_notebook.py). Every cell is plain Python."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Trustworthy RAG — Stage 2 (improvements) on Kaggle
**Settings:** Accelerator = **GPU T4 x2**, Internet = **On**. Add the dataset with the NEW `trag_bundle` (Stage 2 version).
Set `PART` in the first cell, then *Save Version → Save & Run All (Commit)*.
- `PART = "A"` (~4 h): regenerate with token probabilities, improved-retrieval arm, evidence-removal and oracle conditions,
  scoring, leave-one-out faithfulness. Download `trag_results_stage2A.zip`.
- `PART = "B"` (~2 h): train the ModernBERT hallucination checker on RAGTruth, zero-shot DeBERTa-v3-large on RAGTruth,
  apply the checker to all our answers. For B, also **Add Input → the Part A notebook's output** so the new answers get scored.
- `PART = "C"` (~5 h): a third model size, Qwen2.5-14B, with the same 7 systems as Part A (+ trust_rag). Download `trag_results_stage2C.zip`.
- `PART = "D"` (~6 h for 100 questions per domain, unmeasured): ONE large model split over both GPUs, `D_MODEL` = `qwen32b` (Qwen2.5-32B) or
  `gemma27b` (Gemma-3-27B). Reduced sample (`D_LIMIT` questions per domain) and a time budget (`D_BUDGET_H`), so it cannot run into the 12 h
  session cap; generation is resumable (re-run with the previous output attached). Needs nothing from Part C. Two models can run in two
  sessions at the same time on two different Kaggle accounts. Download `trag_results_stage2D_<D_MODEL>.zip`.
- `PART = "E"`: a list of jobs `E_JOBS` (config names, `configs/kaggle_<job>.yaml`) run one after another on ONE Ollama server split over
  both GPUs, under one time budget `E_BUDGET_H`: new model families (`v2_phi4`, `v2_mistral24b`, the Part D subset) and the second financial
  dataset TAT-QA (`tq_<model>`). A job whose model does not fit or does not return log-probs is retried with a smaller KV cache / without
  log-probs instead of stopping the session. Resumable like Part D. Download `trag_results_stage2E_<E_NAME>.zip`.
No API keys are used.""")

code("""PART = "E"   # "E" = job list E_JOBS (kaggle/README.md) ; "A" = generation + scoring + faithfulness ; "B" = checker training + application ; "C" = Qwen2.5-14B ; "D" = one large model over both GPUs
D_MODEL = "qwen32b"   # Part D only: "qwen32b" (Qwen2.5-32B) or "gemma27b" (Gemma-3-27B)
D_LIMIT = 100         # Part D only: questions per domain (100 -> 2,834 answers). The subsample is nested: a later run may raise it and only adds work
D_BUDGET_H = 9.0      # Part D only: stop generating after this many hours (the session cap is 12 h; setup, scoring and zipping need about 1.5 h)
D_KV_Q8 = False       # Part D only: set True ONLY if the check after loading says the model does not fit in GPU memory (halves the KV cache)
D_NO_LOGPROBS = False # Part D only: set True ONLY if the run stopped with "no logprobs" for this model (Gemma); its log-prob analyses are then lost
E_JOBS = ["attr_qwen32b", "attr_llama8b"]   # Part E/F only: config names configs/kaggle_<job>.yaml, in order (Part F = the LLM-as-attributor judges)
E_BUDGET_H = 9.0     # Part E/F only: budget for ALL jobs together (setup, scoring and zipping need about 1-1.5 h of the 12 h cap)
E_NAME = "F"          # Part E/F only: output zip is trag_results_stage2E_<E_NAME>.zip
import glob, os, shutil, subprocess, time, zipfile, pathlib
cands = [p for p in glob.glob("/kaggle/input/**/src", recursive=True) if os.path.isdir(p) and os.path.exists(os.path.join(os.path.dirname(p), "configs"))]
if not cands:   # nothing attached, or the dataset is still processing / was uploaded as something other than trag_bundle.zip
    print("Attached inputs:", sorted(glob.glob("/kaggle/input/*")) or "NONE", "| zip files:", glob.glob("/kaggle/input/**/*.zip", recursive=True)[:5])
    raise SystemExit("The code bundle is not attached. Right panel -> Add Input -> Your Datasets -> the dataset made from trag_bundle.zip, then run again.")
src = next((p for p in cands if "/trag_bundle/" in p), cands[0])   # an attached earlier notebook output also holds a copy of src/ (old code): always take the dataset's
BUNDLE = os.path.dirname(src); WORK = "/kaggle/working/trustworthy-rag"
print("code root:", BUNDLE)
if PART == "B": assert "reference_compile=False" in open(BUNDLE + "/scripts/train_checker.py").read(), "stale code: upload the NEW trag_bundle.zip as a new version of the dataset first"
if PART == "D": assert "--max-hours" in open(BUNDLE + "/scripts/run_generation.py").read() and os.path.exists(f"{BUNDLE}/configs/kaggle_v2_{D_MODEL}.yaml"), "stale code: upload the NEW trag_bundle.zip (Part D version) as a new version of the dataset first"
if PART == "E":
    missing = [j for j in E_JOBS if not os.path.exists(f"{BUNDLE}/configs/kaggle_{j}.yaml")]
    assert not missing, f"no config for {missing}: check the spelling in E_JOBS, or upload the NEW trag_bundle.zip (Part E version)"
    assert os.path.exists(f"{BUNDLE}/data/plans/tatqa.jsonl"), "stale code: upload the NEW trag_bundle.zip (Part E version) as a new dataset first"
    if any(j.startswith("bq_") for j in E_JOBS):
        assert os.path.exists(f"{BUNDLE}/data/plans/bioasq.jsonl"), "stale code: upload the NEW trag_bundle.zip (Part G version) as a new dataset first"
    if any(j.startswith("attr_") for j in E_JOBS):
        assert os.path.exists(f"{BUNDLE}/data/llm_attr/items.jsonl"), "stale code: upload the NEW trag_bundle.zip (Part F version) as a new dataset first"
if not os.path.exists(WORK):
    shutil.copytree(BUNDLE, WORK)
for prev in glob.glob("/kaggle/input/**/experiments/runs/*", recursive=True):   # resume / reuse earlier outputs
    if os.path.isdir(prev) and "/trag_bundle/" not in prev:
        dst = os.path.join(WORK, "experiments/runs", os.path.basename(prev))
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(prev + "/*"):
            if not os.path.exists(os.path.join(dst, os.path.basename(f))): shutil.copy(f, dst)
        print("reused", prev)
assert not os.path.exists(os.path.join(WORK, ".env")), ".env must never be uploaded"
import torch; NGPU = torch.cuda.device_count(); print("GPUs:", NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)])
def sh(cmd, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    print(f"$ {cmd[:100]}\\n{(r.stdout + r.stderr)[-800:]}")
    return r.returncode
def bg(args, log, gpu=None):
    env = dict(os.environ, HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONIOENCODING="utf8")
    if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu % max(NGPU, 1))
    return subprocess.Popen(["python", "-u", "-W", "ignore", "-m", *args], cwd=WORK, env=env,
                            stdout=open(f"/kaggle/working/{log}.log", "w"), stderr=subprocess.STDOUT)
def wait(procs, label, every=300):
    t0 = time.time()
    while any(p.poll() is None for p in procs.values()):
        time.sleep(every)
        for k in procs:
            tail = open(f"/kaggle/working/{k}.log").read().replace(chr(13), chr(10)).strip().splitlines()[-1:] if os.path.exists(f"/kaggle/working/{k}.log") else ""
            print(f"[{label} {(time.time()-t0)/60:.0f} min] {k}: {tail}")
    print(label, {k: p.returncode for k, p in procs.items()})""")

code("""sh('pip -q install "transformers==4.49.0" "sentence-transformers==5.7.0" rank_bm25 pyyaml')
if PART in ("A", "C", "D", "E"):
    import shutil as _s, requests, yaml
    sh("apt-get -qq update && apt-get -qq install -y zstd pciutils")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
    if not _s.which("ollama"):
        sh("curl -fsSL -o /tmp/ollama.tgz https://ollama.com/download/ollama-linux-amd64.tgz && tar -C /usr/local -xzf /tmp/ollama.tgz")
    OLLAMA = _s.which("ollama") or "/usr/local/bin/ollama"
    assert os.path.exists(OLLAMA), "Ollama could not be installed"
    if PART == "D":   # ONE server; Ollama splits the model's layers over both GPUs (a 20 GB model does not fit on one 15 GB T4)
        assert NGPU == 2, f"Part D needs Accelerator = GPU T4 x2 (this session sees {NGPU} GPU(s))"
        D_CFG = f"configs/kaggle_v2_{D_MODEL}.yaml"
        D_CONF = yaml.safe_load(open(f"{WORK}/{D_CFG}"))
        D_PAR = int(D_CONF.get("workers", 3))
        if D_NO_LOGPROBS:   # same run folder and answers, without token log-probabilities
            D_CONF["llm"]["logprobs"] = False
            D_CFG = f"configs/kaggle_v2_{D_MODEL}_nolp.yaml"
            yaml.safe_dump(D_CONF, open(f"{WORK}/{D_CFG}", "w"), sort_keys=False)
        MODELS = [(D_MODEL, D_CONF["llm"]["model"], 11434, None)]
        print("disk free on / (GB):", round(_s.disk_usage("/").free / 1e9), "| the model needs 17-20 GB")
    elif PART == "E":   # Part E starts (and stops) its own server for every job, in the Part E cell below
        assert NGPU == 2, f"Part E needs Accelerator = GPU T4 x2 (this session sees {NGPU} GPU(s))"
        MODELS = []
        print("disk free on / (GB):", round(_s.disk_usage("/").free / 1e9), "| each model needs 1-20 GB; finished models are deleted")
    else:
        MODELS = ([("qwen1.5b", "qwen2.5:1.5b", 11434, 0), ("llama8b", "llama3.1:8b", 11435, 1)] if PART == "A"
                  else [("qwen14b_a", "qwen2.5:14b", 11434, 0), ("qwen14b_b", "qwen2.5:14b", 11435, 1)])
    servers = []
    for tag, model, port, gpu in MODELS:
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_NUM_PARALLEL=str(D_PAR) if PART == "D" else "4",
                   OLLAMA_FLASH_ATTENTION="1", OLLAMA_KEEP_ALIVE="24h")
        if gpu is None:   # Part D: both GPUs visible, always spread the layers
            env.update(CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(NGPU)), OLLAMA_SCHED_SPREAD="1")
            if D_KV_Q8:
                env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu % max(NGPU, 1))
        servers.append(subprocess.Popen([OLLAMA, "serve"], env=env, stdout=open(f"/kaggle/working/ollama_{port}.log", "w"), stderr=subprocess.STDOUT))
    time.sleep(10)
    for tag, model, port, gpu in MODELS:
        pull = subprocess.run([OLLAMA, "pull", model], env=dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}"), capture_output=True, text=True)
        print("pull", model, "-> exit", pull.returncode, (pull.stderr or pull.stdout)[-200:].replace(chr(13), " "))
        body = {"model": model, "stream": False, "logprobs": True, "messages": [{"role": "user", "content": "Reply with the word OK."}]}
        if PART == "D":   # load it exactly as the real requests will (a different num_ctx would make Ollama reload the model)
            body["options"] = {"num_ctx": D_CONF["llm"]["num_ctx"]}
        r = requests.post(f"http://127.0.0.1:{port}/api/chat", timeout=1800, json=body)
        no_lp_ok = PART == "D" and D_NO_LOGPROBS
        assert r.ok and (r.json().get("logprobs") or no_lp_ok), (
            f"{model}: no answer or no logprobs: {r.text[:300]}" + (" (Part D: set D_NO_LOGPROBS = True and run again)" if PART == "D" else ""))
        print(tag, "->", r.json()["message"]["content"][:20], "| logprobs", "ok" if r.json().get("logprobs") else "OFF (D_NO_LOGPROBS)")
    if PART == "D":   # the whole model must sit in GPU memory: layers on the CPU would make generation 10-50x slower
        loaded = requests.get(f"http://127.0.0.1:{MODELS[0][2]}/api/ps").json().get("models", [])
        for m in loaded:
            print(m["name"], "| total GB", round(m["size"] / 1e9, 1), "| in GPU memory GB", round(m["size_vram"] / 1e9, 1))
        sh("nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv")
        assert loaded and all(m["size_vram"] >= 0.97 * m["size"] for m in loaded), (
            "the model is partly on the CPU. Stop, then set D_KV_Q8 = True (or lower workers in the config) and run again")""")

code("""if PART == "A":   # generation with token probabilities (both models in parallel, one per GPU)
    wait({f"gen_{t}": bg(["scripts.run_generation", "--config", f"configs/kaggle_v2_{t}.yaml"], f"gen_{t}") for t in ["qwen1.5b", "llama8b"]}, "generation")""")

code("""if PART == "A":   # scoring + leave-one-out faithfulness (Ollama servers stay up for the regenerations)
    wait({f"score_{t}": bg(["scripts.score_run", "--run", f"v2_{t}", "--loo", "40", "--loo-system", "vanilla_rag"], f"score_{t}", gpu=g)
          for t, g in [("qwen1.5b", 0), ("llama8b", 1)]}, "scoring+LOO")
    for s in servers: s.terminate()""")

code("""if PART == "D":   # one big model over both GPUs: generation with a time budget, then scoring on GPU 0
    import re
    run_dir = f"{WORK}/experiments/runs/v2_{D_MODEL}"
    def n_done():
        return sum(sum(1 for _ in open(f, encoding="utf8")) for f in glob.glob(run_dir + "/*__*.jsonl") if ".scored." not in f)
    dry = subprocess.run(["python", "-m", "scripts.run_generation", "--config", D_CFG, "--limit", str(D_LIMIT), "--dry-run"],
                         cwd=WORK, capture_output=True, text=True)
    m = re.search(r"TOTAL_TODO (\\d+)", dry.stdout)
    assert m, "dry run failed: " + dry.stdout[-500:] + dry.stderr[-500:]
    todo0, n0, t0 = int(m.group(1)), n_done(), time.time()
    print(f"to generate in this session: {todo0} answers ({D_LIMIT} questions per domain), time budget {D_BUDGET_H} h")
    p = bg(["scripts.run_generation", "--config", D_CFG, "--limit", str(D_LIMIT), "--max-hours", str(D_BUDGET_H)], f"gen_{D_MODEL}")
    pause = 60
    while p.poll() is None:
        time.sleep(pause); pause = 600
        h, d = (time.time() - t0) / 3600, n_done() - n0
        rate = d / h if h else 0
        print(f"[D {h * 60:.0f} min] {d}/{todo0} answers | {rate:.0f} per hour" +
              (f" | ETA {(todo0 - d) / rate:.1f} h (budget left {D_BUDGET_H - h:.1f} h)" if rate else ""))
    print("generation exit code", p.returncode, "| generated in this session:", n_done() - n0, "of", todo0)
    for s in servers: s.terminate()
    time.sleep(10)
    wait({f"score_{D_MODEL}": bg(["scripts.score_run", "--run", f"v2_{D_MODEL}"], f"score_{D_MODEL}", gpu=0)}, "scoring-D")""")

code("""if PART == "E":   # jobs in sequence, ONE Ollama server over both GPUs per job, one shared time budget, then scoring
    import re
    URL = "http://127.0.0.1:11434"
    E_T0 = time.time(); E_DEADLINE = E_T0 + E_BUDGET_H * 3600
    CONF = {j: yaml.safe_load(open(f"{WORK}/configs/kaggle_{j}.yaml")) for j in E_JOBS}
    def e_server(parallel, kv_q8=False):
        env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_NUM_PARALLEL=str(parallel), OLLAMA_FLASH_ATTENTION="1",
                   OLLAMA_KEEP_ALIVE="24h", CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(NGPU)), OLLAMA_SCHED_SPREAD="1")
        if kv_q8:
            env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        p = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=open("/kaggle/working/ollama_11434.log", "a"), stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            try:
                requests.get(URL + "/api/version", timeout=5)
                return p
            except Exception:
                pass
        raise RuntimeError("the Ollama server did not start: see ollama_11434.log")
    def e_stop(p):
        p.terminate()
        try:
            p.wait(60)
        except Exception:
            p.kill()
        time.sleep(5)
    def e_load(model, num_ctx):
        # load exactly as the real requests will (a different num_ctx would make Ollama reload); returns (logprobs ok, fully on GPU)
        r = requests.post(URL + "/api/chat", timeout=1800, json={"model": model, "stream": False, "logprobs": True, "options": {"num_ctx": num_ctx},
                                                              "messages": [{"role": "user", "content": "Reply with the word OK."}]})
        if not r.ok:
            print("load failed:", r.text[:300])
            return False, False
        loaded = requests.get(URL + "/api/ps").json().get("models", [])
        for m in loaded:
            print("  ", m["name"], "| total GB", round(m["size"] / 1e9, 1), "| in GPU memory GB", round(m["size_vram"] / 1e9, 1))
        print("   answer:", r.json()["message"]["content"][:20])
        return bool(r.json().get("logprobs")), bool(loaded) and all(m["size_vram"] >= 0.97 * m["size"] for m in loaded)
    def e_script(conf):   # generation jobs (v2_*, tq_*) or the LLM-as-attributor judge (attr_*, Part F)
        return "scripts.llm_attribute" if conf.get("task") == "attribute" else "scripts.run_generation"
    def e_todo(cfg_path):
        dry = subprocess.run(["python", "-m", e_script(yaml.safe_load(open(f"{WORK}/{cfg_path}"))), "--config", cfg_path, "--dry-run"],
                             cwd=WORK, capture_output=True, text=True)
        m = re.search(r"TOTAL_TODO (\\d+)", dry.stdout)
        assert m, "dry run failed: " + dry.stdout[-500:] + dry.stderr[-500:]
        return int(m.group(1))
    def e_done(run):
        return sum(sum(1 for _ in open(f, encoding="utf8")) for f in glob.glob(f"{WORK}/experiments/runs/{run}/*__*.jsonl") if ".scored." not in f)
    STATUS = {}
    for ji, job in enumerate(E_JOBS):
        conf, cfg_path = CONF[job], f"configs/kaggle_{job}.yaml"
        model, run = conf["llm"]["model"], conf["name"]
        todo, left_h = e_todo(cfg_path), (E_DEADLINE - time.time()) / 3600
        print(f"\\n######## job {ji + 1}/{len(E_JOBS)}: {job} ({model}) | {todo} answers to generate | budget left {left_h:.1f} h")
        if todo == 0:
            STATUS[job] = "complete (nothing to do)"; continue
        if left_h < 0.25:
            STATUS[job] = f"NOT STARTED: time budget used ({todo} answers left); rerun with this output attached"; continue
        srv = e_server(conf.get("workers", 3))
        pull = subprocess.run([OLLAMA, "pull", model], env=dict(os.environ, OLLAMA_HOST="127.0.0.1:11434"), capture_output=True, text=True)
        print("pull", model, "-> exit", pull.returncode, (pull.stderr or pull.stdout)[-200:].replace(chr(13), " "))
        if pull.returncode != 0:
            e_stop(srv); STATUS[job] = "FAILED: model download failed (Internet On? tag spelled right?)"; continue
        lp_ok, on_gpu = e_load(model, conf["llm"]["num_ctx"])
        if not on_gpu:   # layers on the CPU would make generation 10-50x slower: retry once with a q8 KV cache (half the memory)
            print("   not fully in GPU memory: restarting with a q8 KV cache")
            e_stop(srv); srv = e_server(conf.get("workers", 3), kv_q8=True)
            lp_ok, on_gpu = e_load(model, conf["llm"]["num_ctx"])
        if not on_gpu:
            e_stop(srv); STATUS[job] = "FAILED: the model does not fit in GPU memory even with a q8 KV cache"; continue
        if not lp_ok:    # same answers and run folder, without token log-probabilities (only the log-prob analyses are lost)
            print("   WARNING: this model returns no log-probs here: generating without them")
            conf = dict(conf, llm=dict(conf["llm"], logprobs=False)); cfg_path = f"configs/kaggle_{job}_nolp.yaml"
            yaml.safe_dump(conf, open(f"{WORK}/{cfg_path}", "w"), sort_keys=False)
        left_h = (E_DEADLINE - time.time()) / 3600
        n0, t0 = e_done(run), time.time()
        p = bg([e_script(conf), "--config", cfg_path, "--max-hours", f"{left_h:.3f}"], f"gen_{job}")
        pause = 60
        while p.poll() is None:
            time.sleep(pause); pause = 600
            h, d = (time.time() - t0) / 3600, e_done(run) - n0
            rate = d / h if h else 0
            print(f"[E {job} {h * 60:.0f} min] {d}/{todo} answers | {rate:.0f} per hour" +
                  (f" | ETA {(todo - d) / rate:.1f} h (budget left {(E_DEADLINE - time.time()) / 3600:.1f} h)" if rate else ""))
        made = e_done(run) - n0
        STATUS[job] = (f"complete ({made} answers, {(time.time() - t0) / 3600:.1f} h)" if made >= todo else
                       f"PARTIAL: {made}/{todo} answers (exit {p.returncode}, see gen_{job}.log); rerun with this output attached")
        if model not in [CONF[j]["llm"]["model"] for j in E_JOBS[ji + 1:]]:   # free the disk for the next models
            subprocess.run([OLLAMA, "rm", model], env=dict(os.environ, OLLAMA_HOST="127.0.0.1:11434"), capture_output=True)
        e_stop(srv)
    runs = sorted({CONF[j]["name"] for j in E_JOBS if CONF[j].get("task") != "attribute" and os.path.isdir(f"{WORK}/experiments/runs/{CONF[j]['name']}")})
    for b in range(0, len(runs), 2):   # NLI scoring, two runs at a time (one per GPU)
        wait({f"score_{r}": bg(["scripts.score_run", "--run", r], f"score_{r}", gpu=i) for i, r in enumerate(runs[b:b + 2])}, "scoring-E")
    print("\\n======== Part E summary (session", round((time.time() - E_T0) / 3600, 1), "h)")
    for j in E_JOBS:
        print(f"  {j:16s} {STATUS.get(j, 'not reached')}")""")

code("""if PART == "C":   # 14B generation (two model copies, domains split), then scoring on both GPUs
    wait({f"gen_{t}": bg(["scripts.run_generation", "--config", f"configs/kaggle_v2_{t}.yaml"], f"gen_{t}") for t in ["qwen14b_a", "qwen14b_b"]}, "generation-14B")
    for s in servers: s.terminate()
    wait({"score_qwen14b": bg(["scripts.score_run", "--run", "v2_qwen14b"], "score_qwen14b", gpu=0)}, "scoring-14B")""")

code("""if PART == "B":   # supervised checker on GPU 0, zero-shot large NLI on GPU 1, in parallel
    wait({"train_checker": bg(["scripts.train_checker", "--model", "answerdotai/ModernBERT-base", "--max-len", "2048", "--epochs", "3"], "train_checker", gpu=0),
          "zs_large": subprocess.Popen(["python", "-u", "-W", "ignore", "-m", "scripts.eval_ragtruth"], cwd=WORK,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(1 % max(NGPU, 1)), NLI_MODEL="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"),
                                       stdout=open("/kaggle/working/zs_large.log", "w"), stderr=subprocess.STDOUT)}, "checkers")
    print(open("/kaggle/working/train_checker.log").read()[-1500:])""")

code("""if PART == "B":   # apply the trained checker to every run we have
    runs = [os.path.basename(p) for p in glob.glob(WORK + "/experiments/runs/*") if os.path.basename(p).startswith(("main_", "v2_"))]
    print("runs:", runs)
    for i, r in enumerate(sorted(runs)):
        sh(f"cd {WORK} && CUDA_VISIBLE_DEVICES={i % max(NGPU, 1)} python -W ignore -m scripts.apply_checker --run {r} --checker ModernBERT-base --systems vanilla_rag trust_rag improved_vanilla improved_trust oracle_vanilla removed_vanilla removed_trust")""")

code("""out = f"/kaggle/working/trag_results_stage2{PART}{'_' + D_MODEL if PART == 'D' else '_' + E_NAME if PART == 'E' else ''}.zip"
def keep(f):
    p = str(f).replace(os.sep, "/")
    if "/checker/" in p and ("/ckpt/" in p or "/model/" in p):
        return False                                  # training checkpoints and the model weights (~600 MB, stay in the notebook output)
    if "/runs/" in p and not f.name.startswith("checker_") and (PART == "B" or "/runs/main_" in p):
        return False                                  # runs are already on the laptop; Part B returns only the new checker scores
    return True
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in pathlib.Path(WORK, "experiments").rglob("*"):
        if f.is_file() and keep(f):
            z.write(f, f.relative_to(WORK))
    for f in glob.glob("/kaggle/working/*.log"):
        name = "logs/" + os.path.basename(f)
        if os.path.basename(f).startswith("ollama_"):
            z.writestr(name, open(f, "rb").read()[-2000000:])   # server chatter can be hundreds of MB: keep the last 2 MB
        else:
            z.write(f, name)
print(out, round(os.path.getsize(out) / 1e6, 1), "MB")""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).parent / "trag_kaggle_stage2.ipynb"
json.dump(nb, open(p, "w", encoding="utf8"), indent=1)
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        compile(c["source"], f"cell{i}", "exec")
print("written", p, "- all code cells compile")
