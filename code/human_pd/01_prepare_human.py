#!/usr/bin/env python3
"""Donor-level GSE243639 microglia from deposited counts and annotations.

Primary selection: published HighLevelCellTypes Micro annotations. Secondary
selection: published Microglia subcluster sheet, without reusing its 10% gene
filter. Both recover all genes from the original deposited count table.
No expression-based selection of target genes. No imputation or integration of
counts. Gzip input is streamed, with memory-bounded parsing across 14 processes.
"""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mg_paths import require_writable_output


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(value):
    return re.sub(r"\.([0-9]+)$", r"-\1", str(value).strip())


def parse_batch(lines, indices, ncols):
    genes, rows = [], []
    for line in lines:
        name, values = line.rstrip().split(",", 1)
        counts = np.fromstring(values, sep=",", dtype=np.float64)
        if len(counts) != ncols or not np.isfinite(counts).all():
            raise ValueError("Invalid count row: " + name)
        selected = counts[indices]
        if np.any(selected < 0) or not np.array_equal(selected, np.rint(selected)):
            raise ValueError("Noninteger or negative source counts: " + name)
        genes.append(next(csv.reader([name]))[0])
        rows.append(sparse.csr_matrix(selected.astype(np.int64)))
    return genes, sparse.vstack(rows, format="csr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    args.out = require_writable_output(args.out)
    if args.out.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.out}")
    args.out.mkdir(parents=True)
    raw = args.raw_dir
    book = raw / "GSE243639_UMAP_coordinates.xlsx"
    clinical_path = raw / "GSE243639_Clinical_data.csv.gz"
    count_path = raw / "GSE243639_Filtered_count_table.csv.gz"
    high = pd.read_excel(book, sheet_name="HighLevelCellTypes")
    final = pd.read_excel(book, sheet_name="Microglia")
    high["cell_id"] = high.CELL_ID.map(canonical)
    final["cell_id"] = final.CELL_ID.map(canonical)
    assert not high.cell_id.duplicated().any()
    assert not final.cell_id.duplicated().any()
    mg = high[high.IDENT.eq("Micro")].copy()
    assert set(final.cell_id) <= set(mg.cell_id)
    mg["donor_id"] = mg.cell_id.str.split("_").str[0]
    mg["final_subcluster"] = mg.cell_id.map(final.set_index("cell_id").IDENT)
    with gzip.open(clinical_path, "rt") as f:
        clinical_text = f.read()
    start = clinical_text.index("N;Brain Bank ID;")
    clinical = pd.read_csv(io.StringIO(clinical_text[start:]), sep=";")
    clinical.columns = clinical.columns.str.strip()
    clinical = clinical.rename(
        columns={
            "Sample ID": "donor_id",
            "Brain Bank ID": "brain_bank_id",
            "Clinical diagnosis": "diagnosis",
            "Age": "age",
            "Sex": "sex",
            "PMI hours": "pmi_hours",
            "RIN measure": "rin",
        }
    )
    assert not clinical.donor_id.duplicated().any()
    assert not clinical.brain_bank_id.duplicated().any()
    clinical["group"] = clinical.diagnosis.map({"Control": "Control", "Parkinson's": "PD"})
    assert clinical.group.notna().all()
    assert set(mg.donor_id) <= set(clinical.donor_id)
    for col in ["age", "pmi_hours", "rin"]:
        clinical[col] = pd.to_numeric(clinical[col], errors="raise")
        assert np.isfinite(clinical[col]).all()
    clinical.to_csv(args.out / "donor_metadata.tsv", sep="\t", index=False)
    mg.to_csv(args.out / "cell_annotation_reconciliation.tsv.gz", sep="\t", index=False)
    blocks, genes = [], []
    with gzip.open(count_path, "rt") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        header = next(csv.reader([next(f)]))[1:]
        cell_ids = pd.Index([canonical(x) for x in header])
        assert not cell_ids.duplicated().any()
        indices = cell_ids.get_indexer(mg.cell_id)
        assert np.all(indices >= 0)
        pending = deque()
        batch = []
        for line in f:
            batch.append(line)
            if len(batch) == 64:
                pending.append(pool.submit(parse_batch, batch, indices, len(header)))
                batch = []
                if len(pending) >= args.workers:
                    g, x = pending.popleft().result()
                    genes.extend(g)
                    blocks.append(x)
                    if len(genes) % 2048 == 0:
                        print(f"Parsed {len(genes)} genes", flush=True)
        if batch:
            pending.append(pool.submit(parse_batch, batch, indices, len(header)))
        while pending:
            g, x = pending.popleft().result()
            genes.extend(g)
            blocks.append(x)
    assert len(genes) == len(set(genes)), "Duplicate gene symbols require explicit resolution"
    x = sparse.vstack(blocks, format="csr")
    sparse.save_npz(args.out / "microglia_raw_counts.npz", x)
    pd.Series(genes, name="gene").to_csv(args.out / "genes.tsv", sep="\t", index=False)
    mg["n_counts"] = np.asarray(x.sum(axis=0)).ravel()
    mg["n_genes"] = np.asarray((x > 0).sum(axis=0)).ravel()
    mito = np.array([g.startswith("MT-") for g in genes])
    mg["mito_counts"] = np.asarray(x[mito].sum(axis=0)).ravel()
    mg.to_csv(args.out / "cell_qc.tsv.gz", sep="\t", index=False)
    donors = pd.Index(clinical.donor_id)
    codes = donors.get_indexer(mg.donor_id)
    assert np.all(codes >= 0)
    panel = pd.read_csv(args.panel, sep="\t")
    # Retain ADRB1 in the additional receptor summaries.
    panel_genes = set(panel.gene.str.upper()) | {"ADRB1"}
    gene_index = pd.Index(genes)
    for selection, keep in [
        ("broad", np.ones(len(mg), dtype=bool)),
        ("final", mg.final_subcluster.notna().to_numpy()),
    ]:
        folder = args.out / selection
        folder.mkdir(exist_ok=True)
        grouping = sparse.csr_matrix(
            (keep.astype(np.int64), (np.arange(len(mg)), codes)), shape=(len(mg), len(donors))
        )
        pb = (x @ grouping).toarray()
        assert np.array_equal(pb.sum(axis=1), np.asarray(x[:, keep].sum(axis=1)).ravel())
        pd.DataFrame(pb, index=pd.Index(genes, name="gene_id"), columns=donors).to_csv(
            folder / "counts.tsv.gz", sep="\t", compression="gzip"
        )
        meta = clinical.copy()
        meta["n_cells"] = np.bincount(codes[keep], minlength=len(donors))
        meta["count_total"] = pb.sum(axis=0)
        meta["detected_genes"] = (pb > 0).sum(axis=0)
        meta["eligible_min20"] = meta.n_cells >= 20
        meta.to_csv(folder / "samples.tsv", sep="\t", index=False)
        rows = []
        for gene in sorted(panel_genes):
            j = gene_index.get_indexer([gene])[0]
            for k, donor in enumerate(donors):
                mask = keep & (codes == k)
                values = x[j, mask].toarray().ravel() if j >= 0 else None
                rows.append(
                    dict(
                        gene=gene,
                        donor_id=donor,
                        group=meta.iloc[k].group,
                        n_cells=int(mask.sum()),
                        measured=j >= 0,
                        positive_cells=int((values > 0).sum()) if j >= 0 else np.nan,
                        prevalence=float((values > 0).mean()) if j >= 0 and mask.any() else np.nan,
                        raw_count_sum=int(values.sum()) if j >= 0 else np.nan,
                    )
                )
        pd.DataFrame(rows).to_csv(folder / "panel_donor_prevalence.tsv", sep="\t", index=False)
    (args.out / "provenance.json").write_text(
        json.dumps(
            dict(
                sources={p.name: sha(p) for p in [book, clinical_path, count_path, args.panel]},
                n_source_cells=len(header),
                n_genes=len(genes),
                n_broad_microglia=len(mg),
                n_final_microglia=len(final),
                workers=args.workers,
                minimum_donor_nuclei=20,
                selection="Author annotations; final-subcluster selection is sensitivity only",
                barcode_adjustment="terminal .1 -> -1 only",
                stage="Braak field measures neurofibrillary tangles, not used as PD stage",
                random_seed="not applicable; deterministic aggregation",
            ),
            indent=2,
        )
        + "\n"
    )
    print("Donor pseudobulks prepared", flush=True)


if __name__ == "__main__":
    main()
