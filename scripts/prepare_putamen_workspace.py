"""Prepare an isolated putamen workspace for annotation or verification.

The copy is deliberate: annotation stages write checkpoints alongside one another.
Never point those stages at the deposited input bundle.
"""

from pathlib import Path
import argparse
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from mg_paths import BUNDLE, OUTPUT, require_writable_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BUNDLE / "xu_putamen")
    parser.add_argument("--out", type=Path, default=OUTPUT / "xu_putamen")
    parser.add_argument("--mode", choices=("annotation", "verify"), default="annotation")
    args = parser.parse_args()
    out = require_writable_output(args.out)
    if out.exists():
        raise FileExistsError(f"Use a fresh workspace: {out}")
    if args.mode == "annotation":
        folders = ["raw"]
    elif args.mode == "verify":
        folders = ["results"]
    else:
        folders = []
    out.mkdir(parents=True)
    for relative in folders:
        shutil.copytree(args.source / relative, out / relative)
    (out / "results").mkdir(exist_ok=True)
    (out / "workspace_manifest.json").write_text(
        json.dumps(
            {"mode": args.mode, "source": str(args.source.resolve()), "input_copies": True},
            indent=2,
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
