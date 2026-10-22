import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGETS = [0, 1, 3, 7, 15, 31]
OUT = "reports/final"
def row(name, d):
    return {"method": name, "ALC": round(d.alc.mean(), 4), **{f"Brier@{n}": round(d[f"brier_{n}"].mean(), 4) for n in BUDGETS}}

b3 = pd.read_csv("reports/phase2_b3/pairs.csv"); h3 = pd.read_csv("reports/phase4/h3_pairs.csv")
ceil = pd.read_csv("reports/phase3/step4b_ceiling_pairs.csv")
rows = [row("B0 constant 0.5", b3[b3.method == "B0_constant_half"]),
        row("B1 official empirical mean", b3[b3.method == "B1_empirical_mean"]),
        row("B2 frozen (submitted)", b3[b3.method == "B2_frozen_m2"]),
        row("B3-global (secondary, benchmark-dependent)", b3[b3.method == "B3_global_prior_ablation"]),
        row("H3 local smoother", h3[h3.method == "H3_local"]),
        row("oracle: B4 with TRUE item difficulty (analysis only, seed 0)", ceil[(ceil.source == "true") & (ceil.lam == 1.0)])]
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/summary_table.csv", index=False); print(tab.to_string(index=False))

fig, ax = plt.subplots(figsize=(6.4, 4))
x = [0, 1, 2, 3, 4, 5]
for r, style in zip(rows, ["k:", "C3-", "C0-", "C2--", "C1--", "C7-."]):
    ax.plot(x, [r[f"Brier@{n}"] for n in BUDGETS], style, marker="o", ms=4, label=r["method"].split(" (")[0])
ax.set_xticks(x); ax.set_xticklabels(BUDGETS); ax.set_xlabel("labels revealed"); ax.set_ylabel("mean Brier over pairs")
ax.set_title("Public pool, 140 pairs, leave-one-benchmark-out"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/learning_curves.png", dpi=160)
