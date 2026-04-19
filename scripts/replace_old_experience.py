r"""Dynamically replace old saved experience with newer experience (V2 requirement).

Directory layout::

    data/replay_buffers/<algo>/<task>/<freshness>/chunk_*.pkl

The script rotates chunks by modification time, keeping at most ``--keep``
files and deleting older ones. Dry-run by default; pass ``--apply`` to
actually remove files.
"""

from __future__ import annotations

import argparse
import glob
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/replay_buffers")
    ap.add_argument("--keep", type=int, default=20)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    for algo in sorted(os.listdir(args.root)) if os.path.isdir(args.root) else []:
        for task in sorted(os.listdir(os.path.join(args.root, algo))):
            for freshness in sorted(os.listdir(os.path.join(args.root, algo, task))):
                d = os.path.join(args.root, algo, task, freshness)
                files = sorted(
                    glob.glob(os.path.join(d, "chunk_*.pkl")),
                    key=os.path.getmtime,
                )
                if len(files) <= args.keep:
                    continue
                to_drop = files[: len(files) - args.keep]
                print(f"[{algo}/{task}/{freshness}] drop {len(to_drop)} chunks")
                for f in to_drop:
                    if args.apply:
                        os.remove(f)
                    else:
                        print("  would remove:", f)


if __name__ == "__main__":
    main()
