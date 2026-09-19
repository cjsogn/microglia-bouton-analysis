# Mouse transcriptomic analyses

Figure 7a and b display the mouse contrasts. Figure S4 retains their intervals. The human cohorts occupy the human PD analysis. Moon temporal profiles and the receptor dot plot occupy Figure 7d and e.

## Reproduce the deposited fits

From the code root:

```sh
python scripts/reproduce_rna_from_counts.py
```

This runner uses the corrected deposited biological-sample count matrices, prepares a fresh output directory, fits all nine mouse contrasts and both existing GSE263094 cell-definition sensitivities, and assembles the panel. It does not recluster the original cells.

Every fit retains expression filtering, TMM normalization, voom, the recorded design and empirical-Bayes moderation. BH adjustment includes all expression-filtered genes within each contrast, not only the displayed panel. The curated panel was not preregistered. The legacy filename `prespecified_gene_panel.tsv` is retained for compatibility.

Independent libraries define pseudobulk columns. Pooled libraries are not converted into fictitious individual-mouse replicates. Shared controls and comparisons within an accession do not constitute independent studies. No cross-study pooled effect is fitted.

## Raw preparation and the RNA import correction

`run_rebuild.py --stage all` runs preparation, differential expression and synthesis. Set `MG_EXTERNAL_DATA` to the directory containing the raw files named in `01_prepare_counts.py`. Use a fresh `MG_OUTPUT_DIR`.

The GSE244652 importer calls official R tximport with `countsFromAbundance="no"`, with no length offset in the retained MARS-seq model. The previous length-scaled import remains a sensitivity. Temporary links named `abundance.h5` allow official reader dispatch without changing the original HDF5 files. Install tximport and rhdf5. Do not substitute an ad hoc HDF5 importer.

`04_audit_import_sensitivity.py` reads the original command metadata, compares both imports and fits the retained leave-one-sample-out checks. Its output must be separate from the input bundle. This import choice is specific to the recorded library design, not a general recommendation for full-length RNA sequencing.

The single-cell preparation script includes a Harmony API orientation adapter and dimension guard. It retains the Harmony algorithm. Numerical cluster IDs require marker review after reclustering, particularly if dependencies change.

See [the validation record](../../docs/VALIDATION.md) for what was rerun in this packaging pass and [environments](../../docs/ENVIRONMENTS.md) for dependencies.
