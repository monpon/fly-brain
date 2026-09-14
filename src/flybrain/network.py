"""Turn a connectome weight matrix into a running Brian2 spiking network.

The connectome fixes *which* neurons connect and how many synapses each pair
shares. It says nothing about membrane dynamics, synaptic strength in volts,
or timing. Everything in `LIFParams` is therefore an assumption layered on top
of the anatomy, following the parameterisation Shiu et al. used for their
whole-brain Drosophila model. Treat these numbers as a starting point to be
varied, not as measured quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from brian2 import (
    Hz,
    Mohm,
    NeuronGroup,
    Network,
    PoissonGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    mV,
    ms,
    prefs,
    second,
)


@dataclass(frozen=True)
class LIFParams:
    """Leaky integrate-and-fire parameters. None of these come from the EM."""

    v_rest: float = -52.0       # mV, resting potential
    v_reset: float = -52.0      # mV, reset after a spike
    v_threshold: float = -45.0  # mV, spike threshold
    tau_m: float = 20.0         # ms, membrane time constant
    refractory: float = 2.2     # ms, absolute refractory period
    epsp_per_synapse: float = 0.275  # mV contributed by one anatomical synapse
    timestep: float = 0.1       # ms, integration step

    # Optional homeostatic scaling. A single fixed EPSP per synapse cannot work
    # across cells whose input counts span three orders of magnitude (Mi1 gets
    # ~8 synapses, HSE ~15,548), so wide-field integrators saturate at the
    # refractory ceiling. Setting this rescales each *postsynaptic* cell's
    # weights so its total input magnitude equals this many mV. It is an extra
    # assumption on top of the anatomy, not something the connectome implies.
    normalize_input_mv: float | None = None


def use_fast_codegen() -> None:
    """Cython codegen if a compiler is available, else fall back to numpy."""
    try:
        prefs.codegen.target = "cython"
    except Exception:
        prefs.codegen.target = "numpy"


def build(
    W: np.ndarray,
    params: LIFParams | None = None,
    *,
    name: str = "brain",
) -> tuple[Network, NeuronGroup, SpikeMonitor]:
    """Build a spiking network from a signed synapse-count matrix.

    W[i, j] is the signed synapse count from neuron i onto neuron j: positive
    for excitatory presynaptic cells, negative for inhibitory ones.
    """
    p = params or LIFParams()
    defaultclock.dt = p.timestep * ms

    eqs = """
    dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
    """
    neurons = NeuronGroup(
        W.shape[0],
        eqs,
        threshold="v > v_threshold",
        reset="v = v_reset",
        refractory=p.refractory * ms,
        method="exact",
        namespace={
            "v_rest": p.v_rest * mV,
            "v_threshold": p.v_threshold * mV,
            "v_reset": p.v_reset * mV,
            "tau_m": p.tau_m * ms,
        },
        name=name,
    )
    neurons.v = p.v_rest * mV

    weights = W.astype(np.float64) * p.epsp_per_synapse

    if p.normalize_input_mv is not None:
        total_in = np.abs(weights).sum(axis=0)
        scale = np.divide(
            p.normalize_input_mv, total_in,
            out=np.ones_like(total_in), where=total_in > 0,
        )
        weights *= scale[np.newaxis, :]

    pre, post = np.nonzero(weights)
    synapses = Synapses(neurons, neurons, "w : volt", on_pre="v_post += w")
    synapses.connect(i=pre, j=post)
    synapses.w = weights[pre, post] * mV

    spikes = SpikeMonitor(neurons)
    net = Network(neurons, synapses, spikes)
    return net, neurons, spikes


def add_stimulus(
    net: Network,
    neurons: NeuronGroup,
    targets: np.ndarray,
    rate: float,
    *,
    drive: float = 20.0,
) -> PoissonGroup:
    """Drive `targets` with Poisson input, standing in for optogenetics.

    `drive` is deliberately far above threshold so each input spike reliably
    evokes one output spike -- the target neurons then fire at approximately
    `rate`, which is what an experimentalist controls.
    """
    source = PoissonGroup(len(targets), rate * Hz)
    injection = Synapses(source, neurons, on_pre=f"v_post += {drive} * mV")
    injection.connect(i=np.arange(len(targets)), j=targets)
    net.add(source, injection)
    return source


def rates_by_type(
    spikes: SpikeMonitor,
    types: np.ndarray,
    duration_s: float,
) -> dict[str, float]:
    """Mean firing rate in Hz for each cell type."""
    counts = np.bincount(spikes.i[:], minlength=len(types))
    out = {}
    for t in sorted(set(types)):
        mask = types == t
        out[t] = float(counts[mask].sum() / (mask.sum() * duration_s))
    return out
