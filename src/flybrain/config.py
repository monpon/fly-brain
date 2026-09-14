"""Environment and connection config.

The data directory used to be a hardcoded mount point on one Linux laptop.
Now it resolves, in order:

    1. $FLYBRAIN_DATA_DIR             an explicit choice, absolute or relative
    2. a `.env` beside the project    for a checkout you are working in
    3. the OS cache directory         for an ordinary `pip install`

Nothing here raises if the data is absent -- `data.py` downloads it on demand.
"""

import os
from pathlib import Path

from . import _platform

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEUPRINT_SERVER = "neuprint.janelia.org"
NEUPRINT_DATASET_DEFAULT = "male-cns:v1.0"
MALE_CNS_VERSION = "male-cns-v1.0"


def _load_dotenv() -> None:
    """Minimal .env loader so a checkout works without extra dependencies.

    Looks beside the package and in the current directory, so it works both
    from a git checkout and from an installed wheel run inside a project.
    """
    for candidate in (PROJECT_ROOT / ".env", Path.cwd() / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


_load_dotenv()


def data_dir() -> Path:
    """Root of the bulk data store. Created if absent, never raises."""
    override = os.environ.get("FLYBRAIN_DATA_DIR", "").strip()
    path = Path(override).expanduser() if override else _platform.cache_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir() -> Path:
    path = data_dir() / MALE_CNS_VERSION
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Derived outputs (weight matrices, extracted circuits)."""
    path = data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir() -> Path:
    """Where scripts write results. Relative to wherever you ran them."""
    path = Path(os.environ.get("FLYBRAIN_OUTPUT_DIR", "output")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def gait_file() -> Path:
    """The tuned gait, preferring a locally retuned one over the shipped one.

    `output/gait.json` is ten minutes of CMA-ES, so a copy ships inside the
    package and the fly walks properly on a fresh install with nothing to
    rebuild. A local `output/gait.json` still wins, because if you have just
    run `flybrain tune-gait` that is the one you meant.
    """
    local = Path("output/gait.json")
    if local.is_file():
        return local
    return Path(__file__).resolve().parent / "_bundled" / "gait.json"


def token() -> str:
    tok = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not tok or tok == "paste_your_token_here":
        raise SystemExit(
            "No NEUPRINT_TOKEN found.\n"
            "  1. Log in at https://neuprint.janelia.org\n"
            "  2. Account icon (top right) -> Account -> copy your Auth Token\n"
            "  3. Put NEUPRINT_TOKEN=... in a .env file, or in the environment\n"
            "\nMost of this package does not need a token -- the bulk tables\n"
            "are public and `flybrain fetch` downloads them without one."
        )
    return tok


def dataset() -> str:
    return os.environ.get("NEUPRINT_DATASET", "").strip() or NEUPRINT_DATASET_DEFAULT


def client():
    """Authenticated neuPrint client."""
    from ._deps import require

    return require("neuprint").Client(
        NEUPRINT_SERVER, dataset=dataset(), token=token()
    )
