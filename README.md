# Microglial bouton-association analyses

Scientific analysis and statistics code for the microglial bouton-association study.
The repository contains the analyses used in the reported results, their required
processing stages, configuration, dependencies, tests and review documentation.
Data and generated results are supplied separately.

## Start here

Use Python 3.12. RNA models also require R, edgeR, limma and statmod. Read
[the environment guide](docs/ENVIRONMENTS.md) for the separate raw-image and
single-cell dependencies. The processed-input reruns do not repeat segmentation
or cell annotation.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export MG_BUNDLE_DIR="/path/to/03_data"
export MG_OUTPUT_DIR="/path/to/new/reproduction_run"
export MG_WORKERS=14
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python -m pytest -q
python scripts/reproduce_core_imaging.py
python scripts/reproduce_rna_from_counts.py
python scripts/reproduce_human_from_counts.py
python scripts/reproduce_morphology_primary.py
python code/analysis/morph3_secondary_features.py --topology-source deposited
python code/analysis/morph4_exploratory_mixed_models.py --check-optimizers
```

Obtain the separate data bundle from the authors. There is no public data-release
identifier yet. Set `MG_BUNDLE_DIR` explicitly in this code-only checkout. Inputs
are protected from output overwrites. Use a fresh output location for each run.
RNA wrappers reject existing result directories. Fourteen workers are used where
supported. Raw-image concurrency must also respect available memory.

Use [the reviewer guide](docs/REVIEWER_GUIDE.md) for the review sequence and
[the analysis map](docs/CURRENT_CODE_SCOPE.md) to locate supporting analyses
and their inputs and outputs. The files
`scripts/validate_reproduction.py` and `docs/VALIDATION.md` describe the
checks against the retained outputs. Numerical reproduction does not establish
that every modelling assumption holds.

## Analysis map

| Module | Purpose |
| --- | --- |
| `upstream_imaging/` | Preprocessing, bouton/Iba1 segmentation, geometric endpoints, spatial null and mask sensitivity |
| `code/morphology_processing/` | Iba1 cell-morphology preprocessing, segmentation and merged-cell splitting |
| `code/analysis/estimate_enrichment.py` | Primary animal-level enrichment and exact tests |
| `code/analysis/estimate_interactions.py`, `timepoint_specificity.py` | Paired marker and temporal specificity |
| `code/analysis/regional_tests.py`, `regional_temporal_contrast.py` | Regional estimates and interactions |
| `code/analysis/morph1_topology_features.py` through `morph4_exploratory_mixed_models.py` | Branch extraction, primary and secondary morphology, exploratory mixed models |
| Other modules in `code/analysis/` | Reported sensitivity, calibration, density, intensity and precision analyses |
| `code/transcriptomics/` | Mouse count preparation, tximport, limma-voom and import/sample checks |
| `code/receptors/` | Mouse microglial selection and descriptive receptor/time-course summaries |
| `code/human_pd/` | Human SNpc donor pseudobulk and covariate-adjusted models |
| `code/xu_putamen/` | Independent putamen annotation, donor models and sensitivity checks |

## Statistical review

Animals are the independent imaging units. Review pooling before the enrichment
ratio, regional weights, exact allocations, confidence-set inversion and separate
multiple-testing families. The exploratory transmitter grouping was selected after
inspection and was not preregistered. The morphology threshold is recomputed in
each allocation of the primary test. The mixed-model screen is a separate,
asymptotic exploratory analysis, with all fit warnings and convergence states saved.

## Bioinformatics review

Review library or donor replication, expression filters, model rank, covariates,
tximport settings and whole-transcriptome correction. Mouse, SNpc and putamen
contrasts have separate FDR families. Adding human data does not change mouse
significance. The pooled Moon libraries support descriptive summaries, not
animal-level treatment tests. No retained model uses imputation.

The shared panel definition is `config/transcript_panel.tsv`: 25 curated genes,
with 12 marked for the compact comparison. All 25 are retained in the result
tables. The reporting subset does not define the multiple-testing family.
See the reviewer guide for result extraction and primary-record metadata checks.

Subsystem READMEs explain upstream stages. For reclustering, inspect identity
markers before reusing cluster identifiers. Count-level reruns use the retained
selections. Raw confocal acquisitions are stored separately and are not bundled.
See [confocal processing](docs/CONFOCAL_PIPELINE.md) and
[analysis scope](docs/ANALYSIS_SCOPE.md) for input requirements and scope limits.

The code is distributed under the MIT license in `LICENSE`.
Keep data, credentials, environments and generated outputs outside version control.

## Check the code

The development dependencies include the raw-image backend needed by the image
processing tests. The tests use synthetic fixtures and do not require study data.

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```
