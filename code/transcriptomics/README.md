# Mouse transcriptomic analyses

Figure 7A displays the LPS contrasts and Figure 7D displays the MPTP contrasts. Figure S3 reports their intervals. Dataset descriptions and the curated panel are in Tables S11 and S15. Import and sample sensitivity results are in Table S17.

## Reproduce the retained fits

From the code root, set `MG_BUNDLE_DIR` and a fresh `MG_OUTPUT_DIR`, then run:

```sh
python scripts/reproduce_rna_from_counts.py
```

The runner stages only the required count matrices, sample metadata, panel definitions and input verification records. It does not copy existing model results or analysis caches. `staged_inputs.json` records the copied inputs and their SHA256 hashes. Manifest paths are rewritten to the new run.

The runner fits the mouse contrasts and the GSE263094 cell-definition sensitivities, then assembles the panel. It uses retained cell selections and does not recluster the source cells. Expression filtering, TMM normalization, voom, design matrices and empirical-Bayes moderation are unchanged. BH adjustment includes all expression-filtered genes within each contrast, not just the displayed panel.

Libraries or biological samples define count-matrix columns. Multiple contrasts from one accession remain separate estimates, not independent studies. No pooled cross-study effect is fitted.

## Prepare counts from the source files

From the code root:

```sh
python code/transcriptomics/run_rebuild.py --stage all
```

Set `MG_EXTERNAL_DATA` to the directory containing the accession-specific inputs named in `01_prepare_counts.py`. Use a fresh output directory. Single-cell preparation requires the additional dependencies in [ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md). Inspect identity markers before reusing cluster identifiers after reclustering.

GSE244652 uses official R tximport with `countsFromAbundance="no"` and no length offset for the MARS-seq primary model. The length-scaled import is a separate sensitivity. Temporary links named `abundance.h5` enable the official HDF5 reader without modifying the quantification files. See [validation](../../docs/VALIDATION.md) for a gene-row-order effect in three length-scaled omission fits and its exact reconciliation.

## Import and sample sensitivity

```sh
python code/transcriptomics/04_audit_import_sensitivity.py --counts-dir /path/to/prepared/transcriptomics --raw-dir /path/to/GSE244652_MPTP_sorted --out /path/to/new/import_sensitivity
```

Use the retained preparation bundle, or a full source-count preparation, containing both `counts.tsv.gz` and `length_scaled_counts_sensitivity.tsv.gz` for the GSE244652 contrasts. The short count-level runner stages only the inputs needed for its own fits.

The sensitivity script reads the original command metadata, compares the two imports and repeats the specified leave-one-sample-out models. Original counts are read directly from `counts.tsv.gz`. All samples remain in the primary fits.

## Input naming

`config/transcript_panel.tsv` is the shared 25-gene definition. Nonmissing `comparison_order` values identify the 12 genes in the comparative display, including Adrb2. These genes cover the processes discussed in the study, while the complete panel remains available in Tables S15–S17. Panel extraction does not change any expression filter, fitted estimate or whole-transcriptome FDR value.

To extract the full panel from the retained complete fits:

```sh
python scripts/export_transcript_panels.py --out /path/to/new/panel_results
```

The output includes all requested genes and explicit measured/tested status. The human models use uppercase human gene symbols. The putamen annotation exclusions are a separate, fixed list in its annotation module, not the reporting catalogue.

The curated panel is stored as `prespecified_gene_panel.tsv` in the supplied data. This input name does not imply preregistration. File and column names used by retained data are kept stable unless the workflow explicitly maps them.

See [validation](../../docs/VALIDATION.md) for executed checks and [analysis scope](../../docs/ANALYSIS_SCOPE.md) for the boundary between retained-input reruns and source processing.
