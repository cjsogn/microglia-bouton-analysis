# Statistical and bioinformatics review

Start with [CURRENT_CODE_SCOPE.md](CURRENT_CODE_SCOPE.md) to locate the analysis behind each current result. Set `MG_BUNDLE_DIR` to the separate `03_data/` bundle. Active inputs are under `coauthor_inputs/`, with cohort-specific inputs in `corrected/` and `xu_putamen/`.

## Reproduction sequence

The README gives the short count- and measurement-level reruns. These reproduce retained selections without repeating segmentation or cell annotation. Use a fresh `MG_OUTPUT_DIR` for each run.

The mouse RNA runner copies only required inputs and records their hashes. For secondary morphology, `--topology-source deposited` explicitly selects retained measurements. Use `--topology-source regenerated` after topology extraction in the same output tree. An explicit missing input stops the analysis.

## Statistical review

Animals are the independent imaging units. Review count pooling before enrichment ratios, equal regional weighting, paired marker comparisons, exact animal-label tests, confidence-set inversion and the stated multiple-testing families. Boutons, stacks and cells are not independent treatment replicates.

The transmitter-class comparison is exploratory. The primary morphology permutation recomputes the control percentile in every allocation. The exploratory mixed-model screen uses the same specified models when checking additional optimizers. Failed or nonconverged fits remain visible in its diagnostics.

The secondary morphology display divides effects by the SD across retained animals. The primary branch-count display uses its specified pairwise SD. Precision intervals from `equivalence_tests.py` are descriptive confidence sets, not equivalence tests.

## Bioinformatics review

Review count-matrix columns, sample and donor metadata, expression filters, model rank, covariates, normalization, transcript import and whole-transcriptome BH correction. Mouse, SNpc and putamen fits have separate contrast-specific families.

The GSE244652 primary model uses original gene counts without a length offset. Its length-scaled import is a sensitivity. The Moon condition libraries support descriptive summaries, not replicated treatment tests. No retained donor model uses imputation.

Subsystem READMEs give the source-processing commands. Reclustering requires inspection of identity markers before reusing cluster identifiers. Target-panel genes are excluded from the independent putamen annotation features.

## Validation limits

[VALIDATION.md](VALIDATION.md) records executed checks. [CONFOCAL_PIPELINE.md](CONFOCAL_PIPELINE.md) identifies retained measurement inputs whose complete upstream generation is not verified in the portable release. Reproducing numbers does not independently validate model assumptions.
