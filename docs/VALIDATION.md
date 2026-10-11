# Validation

## Release checks, 11 October 2026

A fresh Python 3.12 environment installed from `requirements-dev.txt` passed
the test suite, dependency checks, Ruff lint and formatting checks. The tests
cover statistical helpers, input/output separation, RNA input staging, panel
extraction, image geometry, mask processing and explicit morphology input
selection. All five R scripts passed syntax checks.

Fresh runs reproduced core imaging summaries, nine mouse RNA contrasts, two
mouse cell-selection checks, four human SNpc fits and thirteen putamen fits.
All 33 comparison tables, including whole-transcriptome results, matched
retained values at relative tolerance `1e-8` and absolute tolerance `1e-10`.
Three additional morphology tables and the descriptive precision-interval
table matched exactly. The added confidence-grid guard did not change any
retained interval.

The extracted morphology-processing functions matched their original
implementations exactly for eight processing stages on synthetic volumes.
Repeating merged-cell splitting for twelve sampled retained initial masks
reproduced every corresponding final label array exactly. The command-line
test also checks concurrent processing, output hashes and overwrite rejection.
Separately, the retained initial and final masks and the general feature table
were reconciled with their original processing outputs by file hash.

These checks did not repeat preprocessing of every raw acquisition. The mask
workflow and the retained eligible-cell analysis set are distinct. See
[CONFOCAL_PIPELINE.md](CONFOCAL_PIPELINE.md) for their input requirements.

## Transcriptomic panel and metadata checks

The 25-gene catalogue and ordered 12-gene comparison subset share one
configuration. Full-panel extraction covers the nine mouse primary contrasts,
56 MPTP import/sample fits, four SNpc fits and thirteen putamen fits. Statistics
extracted on 4 October matched the retained complete-fit files exactly. The
reporting subset does not define the whole-transcriptome BH family.

On 4 October, fresh GEO SOFT records for all seven accessions and the published
putamen donor table were reconciled with the retained metadata. All 386 field
comparisons passed. They covered sample assignment, diagnosis, tissue and cell
selection, and human age, sex, postmortem interval and SNpc RIN where applicable.
Treatment timing was checked against the source protocols and publication
records. The fixed eleven putamen annotation exclusions were unchanged.

## RNA import and sample-omission checks, 30 September 2026

All four GSE244652 contrasts were reimported from the original Kallisto files
with official tximport. The eight original-count and length-scaled matrices
matched retained values. The run used tximport 1.38.2 and rhdf5 2.54.1.

All 56 import and leave-one-sample-out models completed. Of these, 53 matched
the strict tolerance above. Three length-scaled striatal omission fits had
small numerical differences when identical gene counts were supplied in a
different row order. No whole-transcriptome BH decision at `q < 0.05` changed,
including in the reporting subset.

Repeating those models with the retained gene order reproduced every numerical
column exactly. The affected fits omit GSM7823317, GSM7823295 and GSM7823296.
Retain the input row order when reproducing the corresponding files under
`corrected/mptp_audit/contrasts/`. The comparison tolerance was not relaxed and
retained results were not overwritten. These import checks were not repeated
on 11 October.

## Scope limits

The October release checks did not rerun the complete raw confocal workflow,
the full spatial null, single-cell reannotation, long marker-calibration
simulations or exploratory morphology mixed models. Their retained outputs
are inputs to the relevant downstream checks. The last mixed-model
convergence checks were executed on 19 September.

The complete upstream general-feature extraction and density/brightness
measurement workflows are not verified as portable pipelines in this
repository. Numerical reproduction establishes agreement with retained
outputs, not independent validation of every model assumption.
