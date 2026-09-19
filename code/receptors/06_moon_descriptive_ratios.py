"""Descriptive ratios of mean normalised counts, not mean log values or DE."""

from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mg_paths import OUTPUT, require_writable_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=OUTPUT / "moon_ratios/expression_prevalence.tsv")
    a = p.parse_args()
    destination = require_writable_output(a.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(a.data / "selected_microglia_raw.h5ad", backed="r")
    summary = pd.read_csv(a.data / "expression_prevalence.tsv", sep="\t")
    genes = summary.loc[summary.feature_present, "gene"].drop_duplicates().tolist()
    index = source.var_names.get_indexer(genes)
    if (index < 0).any():
        raise ValueError("Summary contains genes absent from raw features")
    totals = {c: np.zeros(len(genes)) for c in source.obs.condition.unique()}
    sizes = {c: 0 for c in totals}

    def work(job):
        counts, conditions = job
        lib = np.asarray(counts.sum(axis=1)).ravel()
        if np.any(lib <= 0):
            raise ValueError("Empty selected cell")
        norm = sparse.diags(10000 / lib) @ counts[:, index]
        return [
            (c, np.asarray(norm[conditions == c].sum(axis=0)).ravel(), int((conditions == c).sum()))
            for c in np.unique(conditions)
        ]

    # Bounded batches: at most 14 blocks of 512 cells are resident.
    with ThreadPoolExecutor(max_workers=14) as pool:
        for start in range(0, source.n_obs, 14 * 512):
            jobs = []
            for lo in range(start, min(start + 14 * 512, source.n_obs), 512):
                hi = min(lo + 512, source.n_obs)
                jobs.append((source.X[lo:hi].tocsr(), source.obs.condition.iloc[lo:hi].to_numpy()))
            for result in pool.map(work, jobs):
                for c, total, n in result:
                    totals[c] += total
                    sizes[c] += n
    means = {c: pd.Series(totals[c] / sizes[c], index=genes) for c in totals}
    rows = []
    for _, r in summary.iterrows():
        baseline = means["PBS"].get(r.gene, np.nan)
        value = means[r.condition].get(r.gene, np.nan)
        ratio = np.log2(value / baseline) if baseline > 0 and value > 0 else np.nan
        assert sizes[r.condition] == r.selected_microglia
        rows.append(
            dict(
                gene=r.gene,
                condition=r.condition,
                mean_count_10k=value,
                pbs_mean_count_10k=baseline,
                log2_ratio_to_pbs=ratio,
                ratio_status="feature_absent"
                if not r.feature_present
                else ("finite" if np.isfinite(ratio) else "zero_mean_no_pseudocount"),
            )
        )
    ratios = pd.DataFrame(rows)
    summary = summary.drop(
        columns=[c for c in ratios.columns if c not in ["gene", "condition"] and c in summary],
        errors="ignore",
    )
    summary.merge(ratios, on=["gene", "condition"], validate="one_to_one").to_csv(
        destination, sep="\t", index=False
    )
    source.file.close()
    print("Descriptive ratios from full-feature normalisation. No pseudocount, no tests.")


if __name__ == "__main__":
    main()
