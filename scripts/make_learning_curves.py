"""Rebuild report/figures/learning_curves.pdf with embedded TrueType fonts.

Run from the repository root: python make_learning_curves.py OUTPUT.pdf
Reads only the pair-level outputs of the released scripts.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
BUDGETS = [0, 1, 3, 7, 15, 31]
STRICT_M = {"multi_swebench": 3.0, "real_webagents": 2.0, "researchcodebench": 1.5, "swe_rebench": 2.0}

b2 = pd.read_csv("reports/phase2_b2/pairs.csv")
b3 = pd.read_csv("reports/phase2_b3/pairs.csv")
h3 = pd.read_csv("reports/phase4/h3_pairs.csv")
ceil = pd.read_csv("reports/phase3/step4b_ceiling_pairs.csv")
strict = b2[b2.m == b2.bench.map(STRICT_M)]
print("strict rows", len(strict), "benches", strict.bench.unique().tolist())

SERIES = [
    ("B0 constant 0.5", b3[b3.method == "B0_constant_half"], "k", ":", "o"),
    ("B1 empirical mean (official baseline)", b3[b3.method == "B1_empirical_mean"], "#C0392B", "-", "o"),
    (r"B2, strict LOBO ($m^*$ per fold)", strict, "#1F5FA8", "--", "s"),
    (r"B2, fixed $m=2$ (submitted)", b3[b3.method == "B2_frozen_m2"], "#1F5FA8", "-", "o"),
    ("B3-global prior mean", b3[b3.method == "B3_global_prior_ablation"], "#2E8B57", "-.", "^"),
    ("H3 local smoother", h3[h3.method == "H3_local"], "#E67E22", "--", "v"),
    ("Oracle: true item difficulty (analysis only)", ceil[(ceil.source == "true") & (ceil.lam == 1.0)], "#7F7F7F", "-.", "D"),
]
W, H = 376.4737828125, 151.28
fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([41.2830078125 / W, 33.2 / H, 161.82 / W, 110.88 / H])
x = list(range(6))
for label, d, color, ls, marker in SERIES:
    y = [d[f"brier_{n}"].mean() for n in BUDGETS]
    print(f"{label[:30]:30s} n={len(d):4d}", [round(v, 4) for v in y])
    ax.plot(x, y, color=color, ls=ls, marker=marker, ms=3.5, lw=1.2, label=label)
ax.set_xticks(x); ax.set_xticklabels(BUDGETS)
ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30])
ax.tick_params(labelsize=7)
ax.set_xlabel(r"labels revealed $n$ (positions at $\log_2(n+1)$)", fontsize=7.5)
ax.set_ylabel("mean Brier over 140 pairs", fontsize=7.5)
ax.set_ylim(0.07, 0.33)
ax.grid(alpha=0.3)
ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
print("ylim", ax.get_ylim())
fig.savefig(sys.argv[1])
