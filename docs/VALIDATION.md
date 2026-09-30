# Validation, 30 September 2026

## Executed checks

The analysis-only repository passed all 41 tests, the configured lint and format checks, and R syntax checks. Tests cover protected output paths, statistical helpers, RNA input staging and explicit morphology input selection.

Fresh runs reproduced core imaging summaries, nine mouse RNA contrasts, two mouse cell-selection checks, four human SNpc fits and thirteen putamen fits. All 33 primary comparison tables, including whole-transcriptome results, matched retained values at relative tolerance `1e-8` and absolute tolerance `1e-10`.

A further 19 comparisons covered primary and secondary morphology, regional results and the secondary topology-input route. The topology-selection check used a copy of retained measurements in the regenerated-input location. It was not a new feature extraction.

All four GSE244652 contrasts were reimported from the original Kallisto files with official tximport. The eight original-count and length-scaled matrices matched retained values. The initial import attempt could not find tximport. Selecting the existing R library restored tximport 1.38.2 and rhdf5 2.54.1 without changing the method.

The fresh RNA model run contained only required input copies and newly generated results. Input hashes are recorded in `staged_inputs.json`.

## Import-sensitivity reproduction

All 56 import and leave-one-sample-out models completed. Of these, 53 matched the strict tolerance above. Three length-scaled striatal omission fits had small numerical differences when the same gene counts were supplied in a different row order. No whole-transcriptome BH decision at `q < 0.05` changed, including in the displayed gene panel.

Counts in those three fits were identical by gene and sample. Repeating the same model with the retained gene order reproduced every reported numerical column exactly. The order-sensitive cases omit GSM7823317, GSM7823295 and GSM7823296. Retain the input row order when reproducing the corresponding files under `corrected/mptp_audit/contrasts/`. The comparison tolerance was not relaxed, and retained results were not overwritten.

## Scope limits

This pass did not rerun raw confocal processing, the full spatial null, full single-cell reannotation, long marker-calibration simulations or the exploratory morphology mixed models. The mixed-model convergence checks remain those executed on 19 September. Formatting changes did not alter their model specification.

Some upstream measurement-generation stages are not verified as portable pipelines. [CONFOCAL_PIPELINE.md](CONFOCAL_PIPELINE.md) identifies these boundaries. Matching results establishes numerical reproduction, not independent validation of model assumptions.

Detailed comparison records and logs are supplied separately in the full package under `06_validation/code_cleanup_2026-09-30/`.
