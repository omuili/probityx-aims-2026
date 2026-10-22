"""Download the competition-eligible public training tables from measurement-db.

Mirrors the eligibility rule of the official baseline (tools/prepare_data.py):
a benchmark directory is eligible when its benchmarks.parquet reports
response_type == "binary" and granularity == "item". Downloads the four core
tables (benchmarks, subjects, items, response) per eligible directory; traces
are optional and large. Records the dataset revision in data/raw/MANIFEST.json.

Usage: python -m paiec.data.download [--out data/raw] [--traces]
"""
from __future__ import annotations
import argparse, json, os, shutil, concurrent.futures as cf
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "aims-foundations/measurement-db"
CORE = ("benchmarks", "subjects", "items", "response")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--traces", action="store_true", help="also download traces.parquet where present")
    a = ap.parse_args()
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", revision="main")
    files = {s.rfilename for s in info.siblings}
    dirs = sorted({f.rsplit("/", 1)[0] for f in files if f.endswith("/benchmarks.parquet") and "/" not in f.rsplit("/", 1)[0]})
    print(f"revision {info.sha[:12]}; {len(dirs)} benchmark dirs: {dirs}")

    def fetch(name):
        return hf_hub_download(REPO, name, repo_type="dataset", revision=info.sha)

    def eligible(d):
        b = pd.read_parquet(fetch(f"{d}/benchmarks.parquet")).iloc[0]
        return d, bool(b.get("response_type") == "binary" and b.get("granularity") == "item"), dict(b.get("response_type", None) and {"response_type": str(b["response_type"]), "granularity": str(b.get("granularity"))} or {})

    elig, skipped = [], {}
    with cf.ThreadPoolExecutor(8) as ex:
        for d, ok, meta in ex.map(eligible, dirs):
            (elig.append(d) if ok else skipped.__setitem__(d, meta))
    print(f"eligible: {elig}; skipped: {skipped}")

    os.makedirs(a.out, exist_ok=True)
    manifest = {"repo": REPO, "revision": info.sha, "eligible": elig, "skipped": skipped, "tables": {}}
    tables = CORE + (("traces",) if a.traces else ())
    for d in elig:
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
        got = []
        for t in tables:
            src = f"{d}/{t}.parquet"
            if t == "response" and src not in files and f"{d}/responses.parquet" in files:
                src = f"{d}/responses.parquet"
            if src not in files:
                continue
            local = fetch(src)
            dst = os.path.join(a.out, d, f"{t}.parquet")
            shutil.copyfile(local, dst); got.append(t)
            print(f"  {d}/{t}: {os.path.getsize(dst)/1e6:.1f} MB", flush=True)
        manifest["tables"][d] = got
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
