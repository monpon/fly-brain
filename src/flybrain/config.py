"""Environment and connection config."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEUPRINT_SERVER = "neuprint.janelia.org"
NEUPRINT_DATASET_DEFAULT = "male-cns:v1.0"

# Bulk connectome tables live on the external `storage` volume, not in the repo
# and not on the 61 GB root partition. Override with FLYBRAIN_DATA_DIR.
DEFAULT_DATA_DIR = Path("/run/media/monpon/storage/fly-brain-data")
MALE_CNS_VERSION = "male-cns-v1.0"


def _load_dotenv() -> None:
    """Minimal .env loader so scripts work without extra dependencies."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def data_dir() -> Path:
    """Root of the bulk data store, verified to actually be mounted."""
    path = Path(os.environ.get("FLYBRAIN_DATA_DIR", DEFAULT_DATA_DIR))
    if not path.exists():
        raise SystemExit(
            f"Data directory not found: {path}\n"
            "The `storage` volume is a session mount (not in /etc/fstab), so it "
            "may not be mounted yet -- open it once in Files, or set "
            "FLYBRAIN_DATA_DIR in .env to point somewhere else."
        )
    return path


def dataset_dir() -> Path:
    return data_dir() / MALE_CNS_VERSION


def cache_dir() -> Path:
    """Derived outputs (weight matrices, extracted circuits)."""
    path = data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def token() -> str:
    tok = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not tok or tok == "paste_your_token_here":
        raise SystemExit(
            "No NEUPRINT_TOKEN found.\n"
            "  1. Log in at https://neuprint.janelia.org\n"
            "  2. Account icon (top right) -> Account -> copy your Auth Token\n"
            "  3. cp .env.example .env  and paste it in\n"
        )
    return tok


def dataset() -> str:
    return os.environ.get("NEUPRINT_DATASET", "").strip() or NEUPRINT_DATASET_DEFAULT


def client():
    """Authenticated neuPrint client."""
    from neuprint import Client

    return Client(NEUPRINT_SERVER, dataset=dataset(), token=token())
