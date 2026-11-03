"""Rebuild ../deliverables/explainer/EXPLAINER.pdf from EXPLAINER.md (A4, print styling) with headless Chrome or Edge.

python -m scripts.build_explainer_pdf      # needs the `markdown` package (in requirements.txt; installed on this laptop)
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

from src.common import ROOT

SRC = ROOT.parent / "deliverables" / "explainer" / "EXPLAINER.md"
OUT = SRC.with_suffix(".pdf")
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "msedge"]
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1d232b; }
h1 { font-size: 21pt; color: #12324f; margin: 0 0 4pt 0; line-height: 1.2; }
h2 { font-size: 15pt; color: #12324f; border-bottom: 2px solid #12324f; padding-bottom: 3pt; margin-top: 0; break-before: page; }
h3 { font-size: 12pt; color: #1f4e79; margin: 14pt 0 4pt 0; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 18pt; }
li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt 0; font-size: 9pt; }
tr { break-inside: avoid; }
th { background: #e3e8ef; color: #12324f; text-align: left; font-weight: 600; }
th, td { border: 1px solid #b8c2d0; padding: 3pt 5pt; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f6f8fa; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f1f3f5; padding: 0 2pt; color: #7a1f17; }
pre { background: #f6f8fa; border: 1px solid #d5dbe3; border-radius: 4pt; padding: 7pt; font-size: 8.3pt; line-height: 1.35;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; color: #1d232b; padding: 0; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; border-left: 4px solid #1f4e79; background: #eef3f8; color: #12324f; }
hr { border: none; border-top: 1px solid #c9d1db; margin: 10pt 0; }
"""
LIST = re.compile(r"^\s*([-*]|\d+\.)\s+")


def to_html(md_text):
    out, in_code = [], False
    for ln in md_text.split("\n"):   # Python-Markdown needs a blank line before a list that follows a paragraph line
        if ln.startswith("```"):
            in_code = not in_code
        prev = out[-1] if out else ""
        if not in_code and LIST.match(ln) and prev.strip() and not LIST.match(prev) and not prev.startswith((" ", "|", "#", ">")):
            out.append("")
        out.append(ln)
    body = markdown.markdown("\n".join(out), extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace("<h2>Contents</h2>", '<h2 style="break-before: auto">Contents</h2>', 1)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Trustworthy RAG Across Domains: The Complete Explainer'
            f'</title><style>{CSS}</style></head><body>{body}</body></html>')


def main():
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if browser is None:
        raise SystemExit("no Chrome/Edge found: open the HTML in a browser and print it to PDF instead")
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "explainer.html"
        html.write_text(to_html(SRC.read_text(encoding="utf8")), encoding="utf8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT}", html.as_uri()],
                       check=True, capture_output=True)
    print("written", OUT)


if __name__ == "__main__":
    main()
