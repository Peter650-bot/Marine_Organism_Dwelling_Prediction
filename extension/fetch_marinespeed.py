#!/usr/bin/env python3
"""
fetch_marinespeed.py -- download the MarineSPEED V1 benchmark (Bosch et al.
2018, Diversity and Distributions 24:144-157) into extension/data/marinespeed/.

Files (all CSV, gzipped; occurrences.zip holds one csv.gz per species):
    species.csv.gz, traits.csv.gz          species list + ecoregion traits
    occurrences.zip                        presences with 71 environmental values
    random_background.csv.gz               20,000 random background points
    {random,pseudodisc}_{species,background}_5cv_folds.csv.gz
                                           the benchmark's pre-built 5-fold splits

The original host (marinespeed.samuelbosch.com) redirects to the LifeWatch
server below. Each download is size-checked against Content-Length; existing
complete files are skipped. Nothing downloaded is ever executed.
"""

import os
import sys
import time
import urllib.request
import zipfile

BASE = "https://rshiny.lifewatch.be/marinespeed/data/V1/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "marinespeed")
FILES = ["species.csv.gz", "traits.csv.gz", "random_background.csv.gz",
         "random_species_5cv_folds.csv.gz", "random_background_5cv_folds.csv.gz",
         "pseudodisc_species_5cv_folds.csv.gz", "pseudodisc_background_5cv_folds.csv.gz",
         "occurrences.zip"]


def fetch(name):
    path = os.path.join(OUT, name)
    with urllib.request.urlopen(urllib.request.Request(BASE + name, method="HEAD"),
                                timeout=60) as r:
        size = int(r.headers["Content-Length"])
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"[ok]   {name} ({size / 1e6:.1f} MB)")
        return path
    print(f"[get]  {name} ({size / 1e6:.1f} MB) ...", flush=True)
    tmp = path + ".part"
    for attempt in range(50):                  # the server stalls; resume with Range
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if have >= size:
            break
        req = urllib.request.Request(BASE + name, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab") as fh:
                if have and r.status != 206:
                    sys.exit(f"{name}: server ignored the Range request")
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
        except (TimeoutError, OSError) as e:
            print(f"       retry {attempt + 1} at {have / 1e6:.1f} MB ({e})", flush=True)
            time.sleep(5)
    got = os.path.getsize(tmp)
    if got != size:
        sys.exit(f"size mismatch for {name}: {got} != {size}")
    os.replace(tmp, path)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in FILES:
        path = fetch(name)
    occ_dir = os.path.join(OUT, "occurrences")
    if not os.path.isdir(occ_dir):
        with zipfile.ZipFile(path) as z:
            for m in z.namelist():           # only plain csv.gz members, no paths
                base = os.path.basename(m)
                if base.endswith(".csv.gz"):
                    os.makedirs(occ_dir, exist_ok=True)
                    with z.open(m) as src, open(os.path.join(occ_dir, base), "wb") as dst:
                        dst.write(src.read())
    print(f"[done] {len(os.listdir(occ_dir))} species files in {occ_dir}")


if __name__ == "__main__":
    main()
