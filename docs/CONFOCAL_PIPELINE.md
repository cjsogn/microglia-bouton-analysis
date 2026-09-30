# Confocal processing and analysis

Paths in this guide are relative to the scientific code directory.

## Bouton association pipeline

| Stage | Script or module |
| --- | --- |
| Link the original acquisition files to a portable manifest | `scripts/prepare_raw_manifest.py` |
| Process stacks, segment Iba1 and boutons, calculate object geometry | `upstream_imaging/scripts/02_run_uniform_pipeline.py` |
| Preprocessing implementation | `upstream_imaging/vendor/01_preprocessing.py` |
| Iba1 segmentation | `upstream_imaging/vendor/02_microglia_segmentation.py` |
| Bouton segmentation | `upstream_imaging/vendor/03_synapse_segmentation.py` |
| Contact and complete-inclusion geometry | `upstream_imaging/vendor/04_contact_geometry.py` |
| Aggregate stack outputs | `upstream_imaging/scripts/03_aggregate_uniform_results.py` |
| Spatial randomisation | `upstream_imaging/scripts/04_run_torus_null.py` |
| Reported mask and detector sensitivity | `upstream_imaging/scripts/06_mask_and_specification_sensitivity.py` |
| Animal-level enrichment and treatment comparisons | `code/analysis/estimate_enrichment.py` |
| Primary downstream runner | `scripts/reproduce_core_imaging.py` |

The processing configuration is `upstream_imaging/config/reanalysis_config.json`. Shared functions are in `upstream_imaging/scripts/common.py`, with numerical defaults in `upstream_imaging/vendor/00_config.py`. The required raw-image dependencies are listed in `requirements-raw-image.txt`.

Original acquisition files remain on the external drive. Saved Iba1 masks, bouton labels, object tables and stack summaries are in the separate data bundle under `processed_confocal/`. Do not confuse saved processing outputs with original acquisitions.

See [ANALYSIS_SCOPE.md](ANALYSIS_SCOPE.md) for commands, marker selection, output paths and memory-aware use of 14 workers. The validation runs started from saved spatial-null outputs. They did not repeat every raw-stack or randomisation run.

## Cell morphology

| Stage | Script |
| --- | --- |
| Final masks to topology features and terminal-branch barcodes | `code/analysis/morph1_topology_features.py` |
| Primary branch-count, regional and threshold analyses | `code/analysis/morph2_persistent_branches.py` |
| Primary-only statistical runner | `scripts/reproduce_morphology_primary.py` |
| Secondary descriptor comparisons from retained measurements | `code/analysis/morph3_secondary_features.py` |

Final cell masks and morphology measurements are supplied under `coauthor_inputs/morphology/` in the data bundle. The topology script starts from final masks. It is not an implementation of the earlier segmentation and iterative mask-refinement workflow.

## Retained-source limits

The final general morphology feature table is available. Its complete extraction and iterative mask-refinement workflow is not verified as a portable pipeline in this release. The exploratory mixed-model screen is implemented in `code/analysis/morph4_exploratory_mixed_models.py`. Run it with `--check-optimizers` to check convergence under the recorded model specification. This screen is separate from the secondary exact tests.

The density multiplicity correction starts from `coauthor_inputs/pipeline/results/absolute_density_treatment_effects.tsv`. The brightness restrictions use the retained `coauthor_inputs/source_data/S12_brightness_restricted.tsv`. Their upstream generators have not been verified in this release. These are retained-input reproduction boundaries, not evidence that the stored results are incorrect.

Source processing and downstream statistical reruns are distinct entry points. See [ANALYSIS_SCOPE.md](ANALYSIS_SCOPE.md) for explicit morphology input selection and the raw-imaging handoff. Do not describe the short runners as an end-to-end raw-data reconstruction.
