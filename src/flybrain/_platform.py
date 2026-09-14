"""The three things that differ between operating systems, in one place.

Everything else in this package is portable already -- `pathlib` throughout,
no `fork`, no `signal`, no shelling out. What was not portable was a handful
of assumptions scattered across 20 scripts: a hardcoded mount point on one
Linux box, `MUJOCO_GL=glfw` forced in six files, and a worker count tuned to
one machine's core count.

They live here now, resolved at import time, so a script never has to know
what it is running on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"
LINUX = sys.platform.startswith("linux")


def cache_root(app: str = "flybrain") -> Path:
    r"""Where large downloaded data belongs on this OS.

    Hand-rolled rather than taking a dependency on `platformdirs`, because the
    core install promise is "numpy, pandas, pyarrow, nothing else" and this is
    fifteen lines. The conventions followed are the standard ones:

        Windows   %LOCALAPPDATA%\flybrain\Cache
        macOS     ~/Library/Caches/flybrain
        Linux     $XDG_CACHE_HOME/flybrain, else ~/.cache/flybrain
    """
    if WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / app / "Cache"
    if MACOS:
        return Path.home() / "Library" / "Caches" / app
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / app


def gl_backend() -> str | None:
    """The MuJoCo rendering backend to request, or None to let MuJoCo decide.

    Six scripts used to do `os.environ.setdefault("MUJOCO_GL", "glfw")` at
    import. That is right on Linux, where the default would otherwise be EGL
    and EGL fails on the Intel iGPU here (docs/KNOWN_ISSUES.md). It is wrong
    everywhere else: Windows and macOS have exactly one sane backend each
    (WGL and CGL) and MuJoCo already picks it. Forcing glfw there swaps a
    working default for one more thing that can fail to import.

    Headless Linux wants `osmesa`, but this does not switch to it
    automatically: osmesa is a separate system package (`libosmesa6`) and
    guessing wrong swaps a backend that fails to find a display for one that
    fails to import at all. Set MUJOCO_GL=osmesa yourself when you need it.
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]          # an explicit choice wins
    if LINUX:
        return "glfw"
    return None


def headless() -> bool:
    """True when there is no display to open a window on."""
    if WINDOWS or MACOS:
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def setup_rendering() -> None:
    """Configure MuJoCo's GL backend for this machine. Safe to call repeatedly.

    Call before importing `mujoco` or `flygym`, which read MUJOCO_GL at import.
    """
    backend = gl_backend()
    if backend:
        os.environ.setdefault("MUJOCO_GL", backend)
    if LINUX:
        import warnings

        warnings.filterwarnings("ignore", message=".*Wayland.*")


def default_workers() -> int:
    """A sensible process-pool size, rather than one box's core count.

    Leaves a core free for the parent, and caps at 12 because the gait search
    is memory-bound past that -- every worker builds its own MuJoCo model.
    Windows pays much more per worker than Linux (spawn re-imports the whole
    module tree where fork does not), so the cap matters more there.
    """
    return max(1, min(12, (os.cpu_count() or 2) - 1))


def has_compiler() -> bool:
    """Whether Brian2 can use its fast C++/Cython code generation path.

    On Linux this is essentially always true -- gcc is present. On Windows it
    needs MSVC Build Tools, and without them Brian2 silently falls back to
    numpy codegen, which for a 19k-neuron network is the difference between
    "2x slower than real time" and "give up". Silently is the problem, so
    `flybrain doctor` reports it.
    """
    if not WINDOWS:
        return bool(shutil.which("cc") or shutil.which("gcc")
                    or shutil.which("clang"))

    # `cl.exe` is only on PATH inside a Developer Command Prompt, so looking
    # for it reports "no compiler" on a machine that has one. vswhere is how
    # setuptools finds MSVC, and it is installed alongside any Visual Studio
    # or Build Tools release since 2017.
    if os.environ.get("VCINSTALLDIR") or shutil.which("cl"):
        return True
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return False
    try:
        found = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(found.stdout.strip())


def describe() -> str:
    """One line for diagnostics."""
    name = "Windows" if WINDOWS else "macOS" if MACOS else "Linux"
    return (f"{name} / Python {sys.version_info.major}.{sys.version_info.minor} / "
            f"GL {gl_backend() or 'auto'} / {os.cpu_count()} cores / "
            f"compiler {'yes' if has_compiler() else 'no'}")
