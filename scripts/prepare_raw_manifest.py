"""Resolve the deposited raw inventory against explicitly supplied raw-image roots.

This does not guess paths or launch processing. It fails if any selected file is
missing. The inventory includes additional markers outside the reported analyses.
Use --marker repeatedly to select the intended marker population.
"""

from pathlib import Path
import argparse
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from mg_paths import DATA, output_dir, require_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lps-root", type=Path)
    parser.add_argument("--mptp-root", type=Path)
    parser.add_argument("--dataset", choices=["LPS", "MPTP", "ALL"], default="ALL")
    parser.add_argument("--marker", action="append")
    args = parser.parse_args()
    table = pd.read_csv(require_file(DATA / "raw_inventory/raw_stack_inventory.tsv"), sep="\t")
    if args.dataset != "ALL":
        table = table[table.dataset.eq(args.dataset)].copy()
    if args.marker:
        table = table[table.marker.isin(args.marker)].copy()
    if table.empty or table.duplicated(["dataset", "file_id"]).any():
        raise ValueError("Empty or duplicate selected raw inventory")
    roots = {"LPS": args.lps_root, "MPTP": args.mptp_root}
    paths = []
    for row in table.itertuples():
        if roots[row.dataset] is None:
            raise ValueError(f"Raw root required for {row.dataset}")
        base = roots[row.dataset].expanduser().resolve()
        path = (base / row.raw_relative_path).resolve()
        if base not in path.parents:
            raise ValueError("Raw relative path escapes its dataset root")
        require_file(path)
        if path.stat().st_size != row.bytes:
            raise ValueError(f"Raw file size differs from inventory: {row.file_id}")
        paths.append(str(path))
    table["raw_path"] = paths
    dest = output_dir("upstream_imaging", "data", "manifests") / "raw_stack_manifest.tsv"
    table.to_csv(dest, sep="\t", index=False)
    print(dest)


if __name__ == "__main__":
    main()
