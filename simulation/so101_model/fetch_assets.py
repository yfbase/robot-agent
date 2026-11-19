"""Fetch the official SO-101 STL meshes pinned in SOURCE.json.

The meshes (~19 MB) are NOT stored in git; they are downloaded on demand from
the exact upstream commit recorded in SOURCE.json, so a fresh clone stays
small while reproduction remains deterministic. Uses git sparse-checkout
(the raw.githubusercontent.com route is blocked on some networks).

Run:
    uv run python simulation/so101_model/fetch_assets.py
    uv run python simulation/so101_model/fetch_assets.py --force   # re-download
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    source = json.loads((HERE / "SOURCE.json").read_text())
    repo, commit, subpath = source["source_repo"], source["source_commit"], source["source_path"]

    if ASSETS.is_dir() and any(ASSETS.glob("*.stl")) and not args.force:
        print(f"assets already present ({len(list(ASSETS.glob('*.stl')))} STL); nothing to do.")
        return

    with tempfile.TemporaryDirectory(prefix="so101_fetch_") as tmp:
        checkout = Path(tmp) / "repo"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo, str(checkout)],
            check=True,
        )
        # Pin to the recorded commit rather than whatever HEAD is now.
        subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
        subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", subpath], check=True)
        src = checkout / subpath / "assets"
        if not src.is_dir():
            sys.exit(f"upstream {subpath}/assets not found at {commit}")
        ASSETS.mkdir(parents=True, exist_ok=True)
        for stl in sorted(src.glob("*.stl")):
            shutil.copy2(stl, ASSETS / stl.name)
    print(f"fetched {len(list(ASSETS.glob('*.stl')))} STL meshes from {repo}@{commit[:12]} into {ASSETS}")


if __name__ == "__main__":
    main()