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

The final general morphology feature table is available, but its entire upstream extraction workflow has not been reconstructed in this release. The forty-descriptor mixed-model screen was recovered and is implemented separately in `code/analysis/morph4_exploratory_mixed_models.py`. Run it with `--check-optimizers` to retry failed fits under the same model specification. This exploratory screen is distinct from the secondary exact-test analysis above.

Some upstream density and brightness result tables also lack a verified portable generator in this release. Their retained values support the current downstream analyses and figures. These source-code gaps should be resolved or explicitly documented before claiming end-to-end raw-data reproducibility. They do not, by themselves, establish that the retained measurements or results are wrong.
