# ProbityX: Predictive AI Evaluation Competition (NeurIPS 2026)

Beta-Binomial shrinkage for sparse response prediction under a benchmark-level cold start. Final Codabench submission: 946829. Submission source:
`submissions/B2_beta_mean/model.py`. The technical report is submitted separately. Licensed under the MIT License (`LICENSE`).

## Layout
- `paiec/data/` utilities for downloading and loading the competition-eligible tables from the gated `measurement-db` dataset (revision recorded in `data/raw/MANIFEST.json`), constructing platform-shaped `Pair` objects, and generating the data report.
- `paiec/eval/metrics.py` Brier, ECE (10 equal-width bins), ALC by trapezoid on log2(n+1), checked against the published weights.
- `paiec/eval/protocol.py` offline reproduction of test-time adaptation: seeded 50/50 acquisition/evaluation split for pairs with >= 80 items, streamed candidates, nested label sets at 0/1/3/7/15/31, per-pair Brier and ECE, equal-weight averaging.
- `paiec/predictors/` B0 constant, B1 official empirical mean, B2 Beta-Binomial (`smoothed_mean.py`, submitted), B3 configuration prior, B4 item-difficulty posterior, H3 local smoother.
- `paiec/features/` item text statistics and frozen text representations (TF-IDF, MiniLM, bge-base) used only in closed analyses.
- `scripts/` one script per experiment; each writes its report under `reports/`.
- `submissions/B2_beta_mean/model.py` source of the Codabench package (standard library only).
- `reports/` generated experiment outputs; ignored by Git.
- `tests/` 40 unit tests.

## Setup
```
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python numpy==2.4.6 pandas==3.0.6 pyarrow==25.0.1 scipy==1.17.1 scikit-learn==1.9.1 huggingface_hub pytest matplotlib tabulate
.venv/bin/hf auth login                                 # measurement-db is gated
# Fetch the exact dataset revision used in the report. Do not run
# paiec.data.download, which resolves the current main branch.
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("aims-foundations/measurement-db", repo_type="dataset",
    revision="bc8204d811823da849c6686bf124d4ca9f82e4de", local_dir="data/raw",
    allow_patterns=[f"{b}/{t}.parquet" for b in ("multi_swebench",
        "real_webagents", "researchcodebench", "swe_rebench")
        for t in ("benchmarks", "subjects", "items", "response")])
EOF
.venv/bin/python -m paiec.data.report                   # reports/data_report.md
.venv/bin/python -m pytest -q tests
```
Versions used: Python 3.11.16, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1, pyarrow 25.0.1. The embedding experiments
(`phase3_step5_screen.py`, `phase4_h3_local.py`) additionally need `torch` (CPU is enough) and `sentence-transformers`, whose versions were not recorded. The dataset is not included; it must be downloaded as above. Generated outputs under `reports/` and the embedding cache under `data/cache/` are not included and are rebuilt by the scripts below.

## Reproduce the results
Run from the repository root, in this order, because later scripts read earlier outputs.
```
mkdir -p reports/final                               # final_summary.py expects this directory
.venv/bin/python scripts/run_baselines.py            # B0, B1 -> reports/baselines.csv
.venv/bin/python scripts/phase2_b2.py                # B2 LOBO selection of m -> reports/phase2_b2/
.venv/bin/python scripts/phase2_b3.py                # B2 frozen vs B0/B1 and B3 comparators -> reports/phase2_b3/
.venv/bin/python scripts/phase3_step2_difficulty.py  # H1 step 2 -> reports/phase3/step2_*
.venv/bin/python scripts/phase3_step4_b4.py          # H1 step 4 -> reports/phase3/step4_*
.venv/bin/python scripts/phase3_step4b_ceiling.py    # analysis-only oracle -> reports/phase3/step4b_*
.venv/bin/python scripts/phase3_step4c_requirement.py
.venv/bin/python scripts/phase3_step5_screen.py      # representation screening -> reports/phase3/step5_*
.venv/bin/python scripts/phase4_h2_pairlevel.py      # acquisition policies -> reports/phase4/h2_*
.venv/bin/python scripts/phase4_h3_local.py          # local smoother -> reports/phase4/h3_*
.venv/bin/python scripts/final_summary.py            # reports/final/summary_table.csv and learning_curves.png
.venv/bin/python scripts/make_learning_curves.py learning_curves.pdf  # Figure 1 of the report
```
Seeds: split seed 0; candidate-order seeds 0, 1, 2; bootstrap seed 0. Selection statistic everywhere: pooled equal-per-pair ALC
over the 140 held-out pairs under leave-one-benchmark-out.

## Codabench package
```
(cd submissions/B2_beta_mean && zip -j ../../dist/B2_beta_mean.zip model.py)
git clone https://github.com/aims-foundations/paiec_baseline third_party/paiec_baseline
.venv/bin/python third_party/paiec_baseline/check_submission_zip.py dist/B2_beta_mean.zip
```
Uploaded archive SHA-256 `2112ff0f26c20dfa7375dc8dcbfefb4f739d49f6c35989e31340d9317f4e9b5d`; the archive timestamp changes on rebuild,
so compare the extracted `model.py`: SHA-256 `928ea8ca34cfc54780ca0642b6660ca8d2d365ec0a74fe6661afefde22f43054`.
The formative Codabench score is a runtime check only: the formative subset is resampled per submission.
