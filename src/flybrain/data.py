"""Get the bulk connectome tables onto this machine, on any OS.

This used to shell out to `wget -c`, which is not present on Windows and, as
it turns out, not present on plenty of Linux installs either. The resume logic
is the only part of wget that mattered here -- a 1.1 GB file over a domestic
connection gets interrupted -- and an HTTP Range request is six lines.

Downloads land in the OS cache directory by default (see `_platform`), not in
the repo, because they are a gigabyte and because several checkouts should
share one copy.
"""

from __future__ import annotations

import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config

BASE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0"
    "/connectome-data/flat-connectome"
)

# The three tables every model here is built from.
CORE = {
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": 1_100_000_000,
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": 13_000_000,
    "body-neurotransmitters-male-cns-v1.0.feather": 42_000_000,
}

# Synapse-level detail. Only needed for spatial analysis -- a network model
# built from synapse counts does not touch these.
EXTRA = {
    "body-stats-male-cns-v1.0-minconf-0.5.feather": 780_000_000,
    "tbar-neurotransmitters-male-cns-v1.0.feather": 2_700_000_000,
    "syn-partners-male-cns-v1.0-minconf-0.5.feather": 6_800_000_000,
    "syn-points-male-cns-v1.0-minconf-0.5.feather": 12_700_000_000,
}

ALL = {**CORE, **EXTRA}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def _progress(done: int, total: int, started: float, name: str) -> None:
    if not sys.stderr.isatty():
        return
    rate = done / max(time.time() - started, 1e-6)
    if total:
        pct = 100.0 * done / total
        width = 28
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        eta = (total - done) / rate if rate else 0
        tail = f"{pct:5.1f}%  {_human(rate)}/s  eta {eta/60:4.1f}m"
    else:
        bar, tail = "", f"{_human(done)}  {_human(rate)}/s"
    sys.stderr.write(f"\r  {name[:34]:34s} [{bar}] {tail}   ")
    sys.stderr.flush()


def download(url: str, dest: Path, *, expected: int = 0, quiet: bool = False) -> Path:
    """Fetch `url` to `dest`, resuming a partial file if one is there.

    The partial lives at `dest.part` and is only renamed into place once the
    server says the transfer is complete, so an interrupted download can never
    be mistaken for a finished one -- which is the failure mode that makes
    `pyarrow` throw an unreadable error about a corrupt Feather footer.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:        # already complete
            part.replace(dest)
            return dest
        raise

    resuming = response.status == 206
    if have and not resuming:
        have = 0                            # server ignored Range; start over

    remaining = int(response.headers.get("Content-Length") or 0)
    total = (remaining + have) if remaining else expected
    started = time.time()
    mode = "ab" if resuming else "wb"

    with response, open(part, mode) as handle:
        done = have
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            done += len(chunk)
            if not quiet:
                _progress(done, total, started, dest.name)

    if not quiet and sys.stderr.isatty():
        sys.stderr.write("\n")
    part.replace(dest)
    return dest


def fetch(name: str, *, quiet: bool = False) -> Path:
    """Path to one table, downloading it first if this machine lacks it."""
    if name not in ALL:
        raise KeyError(f"unknown table {name!r}; known: {sorted(ALL)}")
    dest = config.dataset_dir() / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    _check_space(ALL[name])
    if not quiet:
        print(f"downloading {name} ({_human(ALL[name])}) "
              f"to {dest.parent}", flush=True)
    return download(f"{BASE}/{name}", dest, expected=ALL[name], quiet=quiet)


def fetch_core(*, quiet: bool = False) -> list[Path]:
    """The three tables everything needs. ~1.1 GB, one time."""
    return [fetch(name, quiet=quiet) for name in CORE]


def have_core() -> bool:
    """Whether the core tables are already on this machine."""
    return all((config.dataset_dir() / name).exists() for name in CORE)


def missing_core() -> list[str]:
    return [n for n in CORE if not (config.dataset_dir() / n).exists()]


def _check_space(needed: int) -> None:
    """Refuse to start a 1 GB download onto a disk that cannot hold it."""
    free = shutil.disk_usage(config.dataset_dir()).free
    if free < needed * 1.1:
        raise SystemExit(
            f"Not enough free space at {config.dataset_dir()}: "
            f"{_human(free)} free, need about {_human(needed * 1.1)}.\n"
            "Set FLYBRAIN_DATA_DIR to a volume with room."
        )


def status() -> str:
    """Human-readable account of what is downloaded and what is not."""
    lines = [f"data directory: {config.dataset_dir()}"]
    for name, size in ALL.items():
        path = config.dataset_dir() / name
        tag = "core" if name in CORE else "extra"
        if path.exists():
            lines.append(f"  [x] {name}  ({_human(path.stat().st_size)}, {tag})")
        else:
            lines.append(f"  [ ] {name}  ({_human(size)}, {tag})")
    return "\n".join(lines)
