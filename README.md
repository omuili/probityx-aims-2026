# ProbityX: Predictive AI Evaluation Competition (NeurIPS 2026)

Beta-Binomial shrinkage for sparse response prediction under a benchmark-level cold start. Final Codabench submission: 946829. Submission source:
`submissions/B2_beta_mean/model.py`. Technical report source:

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
uv pip install --python .venv/bin/python numpy pandas pyarrow scipy scikit-learn huggingface_hub pytest matplotlib tabulate
huggingface-cli login                                  # measurement-db is gated
.venv/bin/python -m paiec.data.download --out data/raw  # writes data/raw/MANIFEST.json with the dataset revision (used: bc8204d8)
.venv/bin/python -m paiec.data.report                   # reports/data_report.md
.venv/bin/python -m pytest -q tests
```
Versions used: Python 3.11.16, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1, pyarrow 25.0.1. The embedding experiments
(`phase3_step5_screen.py`, `phase4_h3_local.py`) additionally need `torch` (CPU is enough) and `sentence-transformers`.

## Reproduce the results
```
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
runtime check only: the formative subset is resampled per submission.
