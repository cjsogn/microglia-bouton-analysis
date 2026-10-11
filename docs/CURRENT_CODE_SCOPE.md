# Analysis map

Paths are under `code/analysis/` unless a different directory is stated. Required helpers and reported sensitivity analyses are part of the scientific code. Generated numerical tables remain outside this repository.

| Result or review task | Entry point | Inputs or principal outputs |
| --- | --- | --- |
| Primary association | `scripts/reproduce_core_imaging.py`, `estimate_enrichment.py` | Stack spatial-null outputs to `unbiased_per_animal.tsv`, `unbiased_baseline.tsv`, `unbiased_treatment_effects.tsv` |
| Direct temporal and three-group tests | `timepoint_contrast.py`, `omnibus_three_group.py` | Animal enrichment to temporal and omnibus results |
| Paired marker specificity | `estimate_interactions.py`, `timepoint_specificity.py` | Paired animal-region estimates to interaction results |
| Exploratory transmitter-class comparison | `baseline_class_contrast.py` | Control-animal marker enrichment |
| Exploratory ordering and calibration | `baseline_ordering_test.py`, `baseline_ordering_calibration.py`, `baseline_ordering_calibration_variants.py`, `baseline_ordering_pairwise_calibration.py`, `assemble_ordering_calibration.py` | Retained null arrays to ordering and calibration results. See `CALIBRATION.md` |
| Shared calibration functions | `exch_sim.py`, `exch_variants.py` | Imports used by the calibration stages |
| Regional estimates | `regional_tests.py`, `regional_temporal_contrast.py` | Animal-region counts to regional summaries, temporal contrasts and profile tests |
| Detector, mask and estimator checks | `analysis_choice_sensitivity.py`, `per_stack_estimator_calibration.py`, `upstream_imaging/scripts/06_mask_and_specification_sensitivity.py` | Alternative detector and mask results, aggregation choices and null calibration |
| Spatial null | `upstream_imaging/scripts/04_run_torus_null.py` | Saved masks and detector labels to observed counts and random-shift distributions |
| Distance and object properties | `distance_profile.py`, `bouton_size_and_enclosure.py` | Multiscale null and object measurements |
| Intensity summaries | `intensity_diagnostic.py`, `intensity_stratified.py` | Retained object measurements to intensity summaries and standardized raw-rate comparisons |
| Density multiplicity correction | `recompute_density_bh.py` | Retained density effects to `density_effects_bh.tsv` |
| Precision and count-model checks | `precision_intervals.py`, `hierarchical_count_model.py`, `hierarchical_count_model_dispersion_sweep.py` | Descriptive confidence sets and count-model point estimates |
| Confidence-set implementation check | `confidence_grid.py` | Boundary and connectedness checks for inversion |
| Morphology mask generation | `code/morphology_processing/run_masks.py` | Explicit acquisition, restored-volume or initial-mask manifest to cell masks |
| Primary morphology | `morph1_topology_features.py`, `morph2_persistent_branches.py`, `scripts/reproduce_morphology_primary.py` | Final masks or retained barcodes to branch-count results |
| Secondary morphology | `morph3_secondary_features.py` | Retained shape and selected topology measurements to exact-test tables |
| Exploratory morphology screen | `morph4_exploratory_mixed_models.py --check-optimizers` | Eligible-cell measurements to model comparisons and fit diagnostics |
| Confocal preprocessing and segmentation | `upstream_imaging/scripts/02_run_uniform_pipeline.py`, `03_aggregate_uniform_results.py`, their shared module and `upstream_imaging/vendor/` | Acquisition manifest to masks, labels, object geometry and stack summaries |
| Mouse RNA | `scripts/reproduce_rna_from_counts.py`, `code/transcriptomics/` | Sample count matrices to whole-transcriptome and panel estimates |
| RNA import and sample checks | `code/transcriptomics/04_audit_import_sensitivity.py` | Original quantification metadata and both count imports |
| Receptors and descriptive LPS time course | `code/receptors/04_moon_timecourse.py`, `05_moon_selection_diagnostics.py`, `06_moon_descriptive_ratios.py` | Selected cells to prevalence, normalized expression and descriptive ratios |
| Human SNpc and putamen | `scripts/reproduce_human_from_counts.py`, `code/human_pd/`, `code/xu_putamen/` | Donor counts to separate cohort models and sensitivity results |
| Putamen annotation | `code/xu_putamen/code/02_audit_source_counts.py`, `03_extract_donor_metadata.R`, `04_independent_annotation.py`, `06_verify_analysis.py`, `08_validate_numeric_backend.py` | Source matrices, donor metadata, annotation evidence and numerical checks |

The short runners do not execute every supporting analysis. Subsystem READMEs give the upstream commands. [ANALYSIS_SCOPE.md](ANALYSIS_SCOPE.md) defines input selection, and [VALIDATION.md](VALIDATION.md) distinguishes executed checks from unverified stages.

Source-data filenames are stable identifiers, including `S20_count_model.tsv`. They do not define statistical families.
