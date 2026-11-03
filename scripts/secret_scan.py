"""Fail if any tracked/staged file contains something that looks like an API key.
Used as git pre-commit hook and before publishing code:  python scripts/secret_scan.py [--all]"""
import re, subprocess, sys
PATTERNS = [r"AIza[0-9A-Za-z_\-]{35}", r"gsk_[0-9A-Za-z]{20,}", r"sk-[0-9A-Za-z_\-]{20,}", r"hf_[0-9A-Za-z]{30,}",
            r"(?i)(api[_-]?key|secret|token)\s*[=:]\s*['\"]?[0-9A-Za-z_\-]{24,}"]
args = ["git", "ls-files"] if "--all" in sys.argv else ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
try:
    files = subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).split()
except (subprocess.CalledProcessError, FileNotFoundError):   # no git repository: scan the code and docs directly
    from pathlib import Path
    skip = ("data", "experiments", "dist", ".venv")
    files = [str(p) for p in Path(".").rglob("*") if p.is_file() and p.parts[0] not in skip
             and p.suffix in (".py", ".yaml", ".yml", ".md", ".json", ".ipynb", ".txt", ".sh", ".csv", ".example")]   # a local .env is fine here
bad = []
for f in files:
    if f.endswith(".env") or "/.env" in f:
        bad.append((f, ".env file must never be committed")); continue
    try:
        txt = open(f, encoding="utf8", errors="ignore").read()
    except OSError:
        continue
    for p in PATTERNS:
        if re.search(p, txt):
            bad.append((f, f"matches {p[:18]}...")); break
for f, why in bad:
    print(f"SECRET SCAN BLOCKED: {f}: {why}")
sys.exit(1 if bad else 0)
