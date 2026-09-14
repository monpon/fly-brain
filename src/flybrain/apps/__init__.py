"""The experiments, as runnable modules.

These were `scripts/*.py` at the repo root, which meant they only worked from
a checkout, with the right working directory, invoked through a path that
spelled the interpreter differently on every OS. Inside the package they ship
with the wheel and run three ways, all equivalent:

    flybrain see --stimulus looming
    python -m flybrain.apps.see --stimulus looming
    python src/flybrain/apps/see.py --stimulus looming      # from a checkout

Each module keeps its own argparse and its own `__main__` block. The CLI runs
them through `runpy`, so dispatching one is exactly the same as running it.
"""
