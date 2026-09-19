"""Validate downloaded 10x matrices, without assigning cells or testing PD effects."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import tarfile

from xu_paths import WORK as ROOT

PANEL = [
    "C3",
    "C3AR1",
    "C1QA",
    "TNF",
    "IL1B",
    "P2RY12",
    "TMEM119",
    "CX3CR1",
    "TREM2",
    "AXL",
    "MERTK",
]


def audit(path):
    prefix = path.name.removesuffix("_matrix.mtx.gz")
    with gzip.open(path.with_name(prefix + "_barcodes.tsv.gz"), "rt") as stream:
        barcodes = [line.strip() for line in stream]
    with gzip.open(path.with_name(prefix + "_features.tsv.gz"), "rt") as stream:
        features = [line.rstrip("\n").split("\t") for line in stream]
    with gzip.open(path, "rt") as stream:
        header = next(stream).strip()
        assert header == "%%MatrixMarket matrix coordinate integer general", header
        line = next(stream)
        while line.startswith("%"):
            line = next(stream)
        n_genes, n_cells, n_entries = map(int, line.split())
        assert n_genes == len(features) and n_cells == len(barcodes)
        assert len(set(barcodes)) == n_cells
        total = 0
        observed = 0
        for line in stream:
            gene, cell, count = map(int, line.split())
            assert 1 <= gene <= n_genes and 1 <= cell <= n_cells and count >= 0
            observed += 1
            total += count
        assert observed == n_entries
    symbols = [row[1] for row in features]
    out = {
        "sample": prefix,
        "genes": n_genes,
        "all_cell_barcodes": n_cells,
        "matrix_entries": n_entries,
        "all_cell_umi_total": total,
    }
    out.update({gene + "_feature_rows": symbols.count(gene) for gene in PANEL})
    return out


def main():
    target = ROOT / "raw/counts"
    target.mkdir(exist_ok=True)
    with tarfile.open(ROOT / "raw/GSE161045_RAW.tar") as archive:
        for member in archive.getmembers():
            assert member.isfile() and Path(member.name).name == member.name
        archive.extractall(target, filter="data")
    paths = sorted(target.glob("*_matrix.mtx.gz"))
    assert len(paths) == 12
    with ProcessPoolExecutor(max_workers=14) as pool:
        rows = list(pool.map(audit, paths))
    with (ROOT / "results/source_matrix_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checksums = []
    for path in sorted((ROOT / "raw").iterdir()):
        if path.is_file() and path.suffix not in [".html", ".js"]:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            checksums.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
    (ROOT / "results/source_checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(
        json.dumps(
            {
                "samples": len(rows),
                "all_cell_barcodes": sum(r["all_cell_barcodes"] for r in rows),
                "scope": "all cell types, not microglia",
                "matrix_validation": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
