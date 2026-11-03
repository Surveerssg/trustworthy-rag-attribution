"""Build dist/trag_bundle.zip for Kaggle/Colab from an explicit allowlist, with a secret scan.
python -m scripts.make_kaggle_bundle"""
import re, zipfile
from pathlib import Path
from src.common import ROOT

ALLOW = ["src/**/*.py", "scripts/*.py", "configs/*.yaml", "tests/*.py", "README.md",
         "data/plans/*.jsonl", "data/processed/*/questions.jsonl", "data/raw/ragtruth/*.jsonl", "data/llm_attr/*.jsonl",
         "data/raw/financebench_document_information.jsonl", "kaggle/*.py",
         "experiments/runs/main_qwen1.5b/*__*.jsonl", "experiments/runs/main_llama8b/*__*.jsonl",
         "experiments/runs/main_qwen1.5b/config.yaml", "experiments/runs/main_llama8b/config.yaml"]
SECRET = re.compile(r"AIza[0-9A-Za-z_\-]{35}|gsk_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z_\-]{20,}|hf_[0-9A-Za-z]{30,}")
files = sorted({p for pat in ALLOW for p in ROOT.glob(pat) if p.is_file() and not p.name.endswith(".scored.jsonl")
                and not p.name.startswith("_")})
assert not any(p.name.startswith(".env") for p in files), ".env in bundle"
for p in files:
    if p.suffix in (".py", ".yaml", ".md") and SECRET.search(p.read_text(encoding="utf8", errors="ignore")):
        raise SystemExit(f"secret-like string in {p}; aborting")
out = ROOT / "dist" / "trag_bundle.zip"
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, Path("trag_bundle") / p.relative_to(ROOT))
print(f"{out}  {len(files)} files  {out.stat().st_size/1e6:.1f} MB")
