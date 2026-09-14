"""Optional dependencies, and a readable error when one is missing.

The core install is numpy, pandas and pyarrow: enough to load a circuit, look
at the wiring, and ask what connects to what. Everything expensive is an
extra, because a package that needs a 2 GB download and a C++ compiler before
`import` succeeds is not one you can hand to somebody.

    pip install flybrain            graph, circuits, analysis
    pip install flybrain[torch]     training and memory
    pip install flybrain[body]      MuJoCo physics, walking, mazes
    pip install flybrain[spiking]   Brian2 LIF simulation
    pip install flybrain[all]       everything

The cost of that split is that a missing piece surfaces at call time rather
than install time, so the message has to say exactly which extra to install.
"""

from __future__ import annotations

import importlib
import importlib.util          # find_spec is not reachable via `import importlib`

# module -> (extra that provides it, what you lose without it)
EXTRAS = {
    "torch": ("torch", "training, memory, and anything gradient-based"),
    "brian2": ("spiking", "leaky integrate-and-fire simulation"),
    "mujoco": ("body", "physics, walking, mazes, and the viewer"),
    "flygym": ("body", "the NeuroMechFly body"),
    "cma": ("body", "gait optimisation"),
    "scipy": ("body", "geodesic odour fields and inverse kinematics"),
    "matplotlib": ("plot", "figures and the pong window"),
    "PIL": ("plot", "loading your own images"),
    "neuprint": ("neuprint", "the authenticated neuPrint query path"),
}


class MissingDependency(ImportError):
    """Raised instead of a bare ImportError, with the install command in it."""


def require(module: str, *, reason: str = ""):
    """Import `module`, or raise with the exact command that fixes it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        extra, provides = EXTRAS.get(module, (module, module))
        what = reason or provides
        raise MissingDependency(
            f"{what} needs `{module}`, which is not installed.\n"
            f"    pip install 'flybrain[{extra}]'"
        ) from exc


def have(module: str) -> bool:
    """Whether an optional dependency is importable, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed() -> dict[str, bool]:
    """Every optional dependency and whether this machine has it."""
    return {name: have(name) for name in EXTRAS}
