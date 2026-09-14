"""Connectome-driven fly brain simulation.

Start here:

    import flybrain as fb

    mb = fb.circuits.mushroom_body()      # bundled, no download
    print(mb.summary())                   # 4,161 neurons, 61,210 synapses

    net = fb.BrainNet(mb)                 # needs flybrain[torch]
    net.fit(X, Y)

Nothing above touches the network or a 1.1 GB download: the circuits worth
starting from ship inside the package. Ask for one that does not
(`fb.circuits.from_types(...)`) and the tables are fetched on demand.

Attribute access is lazy, so `import flybrain` costs a numpy import and
nothing else. Torch, MuJoCo and Brian2 load only when you reach for the thing
that needs them, and say which extra to install if it is absent.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# public name -> module it lives in. Resolved on first access.
_EXPORTS = {
    # circuits and the graph
    "Circuit": "trainable",
    "BrainNet": "trainable",
    "build": "trainable",
    "from_mushroom_body": "trainable",
    "pick_device": "trainable",
    # the mushroom body and its biological learning rule
    "MushroomBody": "mushroom_body",
    "extract": "mushroom_body",
    # vision
    "Retina": "vision",
    "BoxEye": "vision",
    "Binocular": "vision",
    "ColumnInput": "vision",
    "disc": "vision",
    "looming": "vision",
    "grating": "vision",
    "drifting_grating": "vision",
    "flash": "vision",
    # checkpoints
    "save": "checkpoint",
    "load": "checkpoint",
    "inspect": "checkpoint",
}

# submodules reachable as attributes, e.g. fb.circuits, fb.data
_SUBMODULES = (
    "circuits", "config", "data", "trainable", "vision", "checkpoint",
    "mushroom_body", "learning", "tasks", "odors", "local_data", "connectome",
    "network", "olfaction", "olfactory_brain", "navigate", "odor_field",
    "body", "cpg", "vnc", "embodied", "maze", "ik", "fitness", "antennae",
    "view", "apps",
)

__all__ = [*sorted(_EXPORTS), *sorted(_SUBMODULES), "__version__"]

if TYPE_CHECKING:                      # so editors and type checkers see them
    from . import circuits, config, data
    from .trainable import BrainNet, Circuit, build


def __getattr__(name: str):
    """PEP 562 lazy export.

    The point is that `import flybrain` must not import torch. A user with the
    core install only should be able to load a circuit and look at it; if the
    package imported everything eagerly, that install would fail at `import`
    rather than at the one call that actually needs the missing piece.
    """
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


def doctor() -> None:
    """Print what this machine has and what it is missing. Same as the CLI."""
    from .cli import cmd_doctor

    cmd_doctor([])
