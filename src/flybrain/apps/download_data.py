#!/usr/bin/env python
"""Download the male-CNS bulk tables.

    flybrain fetch                 # core tables (~1.1 GB)
    flybrain fetch --all           # everything (~24 GB)
    flybrain fetch --status        # what is already here

Kept as a module because `flybrain download-data` was the old name and muscle
memory is real, but the work happens in `flybrain.data`, which downloads over
plain HTTP with a Range header instead of shelling out to `wget`. That matters
more than it sounds: `wget` is absent on Windows and on plenty of Linux images,
and it was the last hard dependency on an external binary anywhere in here.

You usually do not need this at all. The circuits worth starting from are
bundled (`flybrain circuits`), and anything else downloads on first use.
"""

import argparse
import sys

from flybrain import data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flybrain fetch")
    ap.add_argument("--all", action="store_true",
                    help="also the synapse-level tables (~24 GB total)")
    ap.add_argument("--status", action="store_true",
                    help="report what is downloaded, download nothing")
    args = ap.parse_args(argv)

    if args.status:
        print(data.status())
        return 0

    for name in (data.ALL if args.all else data.CORE):
        data.fetch(name)

    print("\n" + data.status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
