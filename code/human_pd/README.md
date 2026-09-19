# Human SNpc microglia, GSE243639

The SNpc cohort and the separately annotated putamen cohort are displayed in the human PD analysis. Neither is pooled with the other.

## From deposited donor counts

```sh
python scripts/reproduce_human_from_counts.py
```

This is the tested short route from the code root. It reproduces the retained donor models for both cohorts.

## From the source matrix and published annotation

```sh
python code/human_pd/01_prepare_human.py --raw-dir /path/to/GSE243639 --panel /path/to/curated_gene_panel.tsv --out /path/to/new/human_prepared --workers 14
Rscript code/human_pd/02_human_limma.R /path/to/new/human_prepared /path/to/new/human_models
```

Both destinations must be fresh. The model output is outside the preparation input. Required source files are the deposited count CSV, annotation workbook and clinical CSV. Preparation records SHA256 values, reads the workbook without alteration and streams the count matrix through bounded worker batches.

Selection uses published microglial labels with exact barcode and donor reconciliation. Counts are summed within donors. Nuclei are not independent disease replicates.

The fixed donor model includes age, sex, postmortem interval, RIN and diagnosis. There is no imputation, covariate selection or fallback model. Missing metadata or a rank-deficient design stops execution. BH correction includes every expression-filtered gene within each fit. The displayed panel does not define the correction denominator.

Retained checks include the alternative annotation, the stated cell-count thresholds and the age-overlap selection. Read each inclusion table alongside its model design. An unchanged donor set under a stricter threshold is not an independent perturbation. Tau Braak stages are not PD stages.


See [the repository overview](../../README.md), [environments](../../docs/ENVIRONMENTS.md) and [validation](../../docs/VALIDATION.md).
