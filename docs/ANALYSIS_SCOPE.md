# Analysis scope

The [repository overview](../README.md) is the current entry-point map. Short reruns start from deposited spatial-null results, eligible-cell features/barcodes and biological-sample count matrices. They do not claim an end-to-end reconstruction from every acquisition file.

## Raw confocal processing

Original acquisition files remain on the external drive and are not copied into the package. Saved masks, detector label arrays and object outputs are in `03_data/processed_confocal/`. The inventory also lists a historical LPS marker outside the manuscript, so select manuscript markers explicitly.

From the code root:

```sh
python -m pip install -r requirements-raw-image.txt
python scripts/prepare_raw_manifest.py --dataset LPS --lps-root /path/to/original/LPS --marker TH --marker ChaT --marker vglut1 --marker vglut2 --marker vgat --marker gad65
python upstream_imaging/scripts/02_run_uniform_pipeline.py --dataset LPS --marker TH --specification uniform_standard --workers 14
python upstream_imaging/scripts/03_aggregate_uniform_results.py
python upstream_imaging/scripts/04_run_torus_null.py --dataset LPS --marker TH --specification uniform_standard --workers 14
```

For MPTP, prepare a separate fresh output directory and use `--dataset MPTP --mptp-root /path/to/original/MPTP --marker TH --marker vglut1`. Review each script's `--help` before running all markers and specifications.

`06_mask_and_specification_sensitivity.py` retains the mask and detector comparisons used in Figure S2. The separate candidate-matching analysis, which is not reported in the current manuscript or supplement, is not included. `MG_RAW_MANIFEST` can select a different explicitly prepared manifest.

Raw preprocessing requires the recorded OpenCV top-hat backend. The portable loader stops if that backend is unavailable, instead of silently choosing the vendor module's alternative implementation.

Outputs go to `MG_OUTPUT_DIR/upstream_imaging`. Use one numerical-library thread per worker and up to 14 workers. Establish memory demand with a representative stack before processing large stacks concurrently. Do not change detector settings, shift sampling or seeds to work around resource limits.

## Morphology

`morph1_topology_features.py` starts from supplied final label masks. `morph2_persistent_branches.py` uses barcodes and the final eligible-cell table and includes primary, regional, density-adjusted and threshold-sweep analyses. `reproduce_morphology_primary.py` is an explicit primary-only runner using the same functions. `morph3_secondary_features.py` assembles retained secondary tests from deposited feature measurements.

The primary permutation recomputes the control percentile in each allocation. The fixed-threshold result remains in an explicitly labelled conditional column. The full-procedure result was already used in the manuscript. Packaging restored it to a stale source-data table and recovered builder.

Earlier acquisition preprocessing, iterative cell-mask refinement and extraction of the general feature table are not presented as a newly reconstructed automated pipeline. Final masks, measurements, inclusion ledgers and provenance are supplied. Distinguish reviewing the origin of measurements from rerunning their statistical analysis.

## Longer analyses

The analysis directory also contains marker-ordering calibration, spatial-null diagnostics, estimator choices, precision intervals and hierarchical count-model sensitivities. Their docstrings identify assumptions and inputs. They support reported analyses or their audit, not alternative primary models.

`derived_input()` prefers a regenerated predecessor in the active output directory and otherwise reads the deposited table. Keep each run in a fresh output directory. Software checks do not replace review of model assumptions.
