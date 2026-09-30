# Analysis map

This map uses the current seven supplementary figures and seventeen supplementary tables. Paths are under `code/analysis/` unless a different directory is stated. Required helpers and reported sensitivity analyses are part of the scientific code.

| Result or review task | Entry point | Inputs or principal outputs |
| --- | --- | --- |
| Primary association, Figures 2, 3 and 6, Tables S3 and S5 | `scripts/reproduce_core_imaging.py`, `estimate_enrichment.py` | Stack spatial-null outputs to `unbiased_per_animal.tsv`, `unbiased_baseline.tsv`, `unbiased_treatment_effects.tsv` |
| Direct temporal and three-group tests | `timepoint_contrast.py`, `omnibus_three_group.py` | Animal enrichment to temporal and omnibus results |
| Paired marker specificity, Tables S6 and S12 | `estimate_interactions.py`, `timepoint_specificity.py` | Paired animal-region estimates to interaction results |
| Exploratory transmitter-class comparison, Figures 2 and 5 | `baseline_class_contrast.py` | Control-animal marker enrichment |
| Exploratory ordering and calibration, Table S13 | `baseline_ordering_test.py`, `baseline_ordering_calibration.py`, `baseline_ordering_calibration_variants.py`, `baseline_ordering_pairwise_calibration.py`, `assemble_ordering_calibration.py` | Retained null arrays to ordering and calibration results. See `CALIBRATION.md` |
| Shared calibration functions | `exch_sim.py`, `exch_variants.py` | Imports used by the calibration stages |
| Regional estimates, Figure 5 and supplementary methods | `regional_tests.py`, `regional_temporal_contrast.py` | Animal-region counts to regional summaries, temporal contrasts and profile tests |
| Detector, mask and estimator checks, Figure S1 | `analysis_choice_sensitivity.py`, `per_stack_estimator_calibration.py`, `upstream_imaging/scripts/06_mask_and_specification_sensitivity.py` | Alternative detector and mask results, aggregation choices and null calibration |
| Spatial null, Figure S2 | `upstream_imaging/scripts/04_run_torus_null.py` | Saved masks and detector labels to observed counts and random-shift distributions |
| Distance and object properties, Figure S5 and Table S14 | `distance_profile.py`, `bouton_size_and_enclosure.py` | Multiscale null and object measurements |
| Intensity summaries, Table S8 | `intensity_diagnostic.py`, `intensity_stratified.py` | Retained object measurements to intensity summaries and standardized raw-rate comparisons |
| Density multiplicity correction, Table S7 | `recompute_density_bh.py` | Retained density effects to `density_effects_bh.tsv` |
| Precision and count-model checks, Tables S9 and S10 | `equivalence_tests.py`, `hierarchical_count_model.py`, `hierarchical_count_model_dispersion_sweep.py` | Descriptive confidence sets and count-model point estimates |
| Confidence-set implementation check | `confidence_grid.py` | Boundary and connectedness checks for inversion |
| Primary morphology, Figures 4 and S6 | `morph1_topology_features.py`, `morph2_persistent_branches.py`, `scripts/reproduce_morphology_primary.py` | Final masks or retained barcodes to branch-count results |
| Secondary morphology, Figure 4d | `morph3_secondary_features.py` | Retained shape and selected topology measurements to exact-test tables |
| Exploratory morphology screen | `morph4_exploratory_mixed_models.py --check-optimizers` | Eligible-cell measurements to model comparisons and fit diagnostics |
| Confocal preprocessing and segmentation | `upstream_imaging/scripts/02_run_uniform_pipeline.py`, `03_aggregate_uniform_results.py`, their shared module and `upstream_imaging/vendor/` | Acquisition manifest to masks, labels, object geometry and stack summaries |
| Mouse RNA, Figures 7A/D and S3, Tables S11 and S15 | `scripts/reproduce_rna_from_counts.py`, `code/transcriptomics/` | Sample count matrices to whole-transcriptome and panel estimates |
| RNA import and sample checks, Table S17 | `code/transcriptomics/04_audit_import_sensitivity.py` | Original quantification metadata and both count imports |
| Receptors and descriptive LPS time course, Figures 2, 7B/C and S4 | `code/receptors/04_moon_timecourse.py`, `05_moon_selection_diagnostics.py`, `06_moon_descriptive_ratios.py` | Selected cells to prevalence, normalized expression and descriptive ratios |
| Human SNpc and putamen, Figure 7E, Tables S16 and S17 | `scripts/reproduce_human_from_counts.py`, `code/human_pd/`, `code/xu_putamen/` | Donor counts to separate cohort models and sensitivity results |
| Putamen annotation, Figure S7 | `code/xu_putamen/code/02_audit_source_counts.py`, `03_extract_donor_metadata.R`, `04_independent_annotation.py`, `06_verify_analysis.py`, `08_validate_numeric_backend.py` | Source matrices, donor metadata, annotation evidence and numerical checks |

The short runners do not execute every supporting analysis. Subsystem READMEs give the upstream commands. [ANALYSIS_SCOPE.md](ANALYSIS_SCOPE.md) defines input selection, and [VALIDATION.md](VALIDATION.md) distinguishes executed checks from unverified stages.

Source-data filenames are identifiers and need not match current table numbers. For example, Table S10 reads `S20_count_model.tsv`. Renumbering the supplement does not change the underlying data or statistical families.
