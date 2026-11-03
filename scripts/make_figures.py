"""Paper figures + tables that are already final (retrieval). Palette: validated categorical slots 1-5
(dataviz reference palette; CVD-checked). Light colours fail 3:1 contrast -> every bar group carries a value
label on its best variant, and each figure has a companion table in paper/tables/.

python -m scripts.make_figures
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import PAPER, RESULTS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed order, never cycled
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DOMS = ["hotpotqa", "squad2", "pubmedqa", "cuad", "financebench"]
DLAB = {"hotpotqa": "General\n(HotpotQA)", "squad2": "General\n(SQuAD 2.0)", "pubmedqa": "Medical\n(PubMedQA)",
        "cuad": "Legal\n(CUAD)", "financebench": "Financial\n(FinanceBench)"}
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def _axes(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_retrieval_variants():
    df = pd.read_csv(RESULTS / "retrieval_variants.csv")
    modes = ["dense", "bm25", "hybrid", "rerank", "scoped"]
    names = {"dense": "Dense (MiniLM) — baseline", "bm25": "BM25", "hybrid": "Hybrid (RRF)",
             "rerank": "Hybrid + cross-encoder rerank", "scoped": "Entity-scoped"}
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    w = 0.16
    for i, m in enumerate(modes):
        sub = df[df["mode"] == m].set_index("domain").reindex(DOMS)
        x = np.arange(len(DOMS)) + (i - 2) * w
        ax.bar(x, sub["hit"], w - 0.02, color=SERIES[i], label=names[m], edgecolor=SURF, linewidth=1)
    for j, d in enumerate(DOMS):                       # label only the best variant per domain
        s = df[df.domain == d].set_index("mode")["hit"]
        best = s.idxmax(); i = modes.index(best)
        ax.text(j + (i - 2) * w, s[best] + 0.02, f"{s[best]:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(DOMS)), [DLAB[d] for d in DOMS])
    ax.set_ylim(0, 1.12); ax.set_ylabel("Hit@5 (gold passage in top 5)")
    _axes(ax)
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "retrieval_variants_hit5.png", dpi=300)
    t = df.pivot(index="domain", columns="mode", values="hit").reindex(DOMS)[modes]
    (PAPER / "tables" / "retrieval_variants_hit5.md").write_text(t.round(3).to_markdown(), encoding="utf8")


def fig_recall_at_k():
    df = pd.read_csv(RESULTS / "retrieval_all.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ends = []
    for i, d in enumerate(DOMS):
        s = df[df.domain == d].sort_values("k")
        ax.plot(s.k, s.chunk_hit, color=SERIES[i], lw=2, marker="o", ms=5, mec=SURF, mew=1.5,
                solid_capstyle="round", label=DLAB[d].replace(chr(10), " "))
        ends.append([s.chunk_hit.iloc[-1], d, i])
    # de-collide end labels: keep >= 0.065 apart (top-down), draw a hairline leader to the true end point
    ends.sort(reverse=True)
    ys = []
    for v, d, i in ends:
        y = min(v, ys[-1] - 0.065) if ys else v
        ys.append(y)
        ax.annotate(DLAB[d].replace(chr(10), " "), xy=(20, v), xytext=(21.5, y),
                    va="center", fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=GRID, lw=1))
    ax.set_xticks([5, 10, 20]); ax.set_xlim(4, 33); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(0.66, 0.0))
    ax.set_xlabel("k (retrieved passages)"); ax.set_ylabel("Hit@k (dense retrieval)")
    _axes(ax)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "retrieval_hit_at_k.png", dpi=300)
    t = df.pivot(index="domain", columns="k", values="chunk_hit").reindex(DOMS)
    (PAPER / "tables" / "retrieval_hit_at_k.md").write_text(t.round(3).to_markdown(), encoding="utf8")


def table_corpus():
    import json
    s = json.load(open(RESULTS / "corpus_stats.json"))
    rows = [{"domain": d, "documents": s[d]["docs"], "chunks": s[d]["chunks"], "avg words/chunk": round(s[d]["avg_chunk_words"]),
             "questions": s[d]["questions"], "answerable": s[d]["answerable"],
             "gold chunk located": f"{s[d]['answerable_with_gold_chunk']}/{s[d]['answerable']}"} for d in DOMS]
    (PAPER / "tables" / "corpus_stats.md").write_text(pd.DataFrame(rows).to_markdown(index=False), encoding="utf8")


if __name__ == "__main__":
    (PAPER / "figures").mkdir(parents=True, exist_ok=True); (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    fig_retrieval_variants(); fig_recall_at_k(); table_corpus()
    print("ok")
