#!/usr/bin/env python
"""Download the male-CNS bulk tables to the storage volume.

    .venv/bin/python scripts/download_data.py           # core tables (~1.1 GB)
    .venv/bin/python scripts/download_data.py --all     # everything (~24 GB)

Resumable: re-running skips files that are already complete.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flybrain import config

BASE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0"
    "/connectome-data/flat-connectome"
)

CORE = {
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": "1.1 GB",
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": "13 MB",
    "body-neurotransmitters-male-cns-v1.0.feather": "42 MB",
}

# Synapse-level detail. Only needed for spatial analysis -- a network model
# built from synapse counts does not touch these.
EXTRA = {
    "body-stats-male-cns-v1.0-minconf-0.5.feather": "780 MB",
    "tbar-neurotransmitters-male-cns-v1.0.feather": "2.7 GB",
    "syn-partners-male-cns-v1.0-minconf-0.5.feather": "6.8 GB",
    "syn-points-male-cns-v1.0-minconf-0.5.feather": "12.7 GB",
}


def main(include_extra: bool) -> None:
    dest = config.dataset_dir()
    dest.mkdir(parents=True, exist_ok=True)

    targets = dict(CORE)
    if include_extra:
        targets.update(EXTRA)

    print(f"destination: {dest}\n")
    for name, size in targets.items():
        print(f"{name}  ({size})")
        subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "--progress=dot:giga",
             f"{BASE}/{name}", "-O", str(dest / name)],
            check=True,
        )

    print("\ndone:")
    for path in sorted(dest.glob("*.feather")):
        print(f"  {path.stat().st_size / 1e9:6.2f} GB  {path.name}")


if __name__ == "__main__":
    main(include_extra="--all" in sys.argv)
