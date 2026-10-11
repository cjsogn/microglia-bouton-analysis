# Analysis scope

Short reruns start from retained spatial-null outputs, eligible-cell measurements and biological-sample count matrices. The [analysis map](CURRENT_CODE_SCOPE.md) links these stages to the current results. [CONFOCAL_PIPELINE.md](CONFOCAL_PIPELINE.md) identifies the upstream processing stages and remaining source limits.

## Confocal processing

Original acquisitions remain on the external drive. Saved masks, detector labels and object outputs are supplied under `03_data/processed_confocal/`. Select the reported markers explicitly when preparing the acquisition manifest.

From the code root:

```sh
python -m pip install -r requirements-raw-image.txt
python scripts/prepare_raw_manifest.py --dataset LPS --lps-root /path/to/original/LPS --marker TH --marker ChaT --marker vglut1 --marker vglut2 --marker vgat --marker gad65
python upstream_imaging/scripts/02_run_uniform_pipeline.py --dataset LPS --marker TH --specification uniform_standard --workers 14
python upstream_imaging/scripts/03_aggregate_uniform_results.py
python upstream_imaging/scripts/04_run_torus_null.py --dataset LPS --marker TH --specification uniform_standard --workers 14
```

This example processes TH. Repeat the relevant stages for the other selected markers and required specifications. MPTP processing uses a separate fresh output directory and `--dataset MPTP --mptp-root /path/to/original/MPTP --marker TH --marker vglut1` when preparing the manifest.

`06_mask_and_specification_sensitivity.py` implements the detector and mask sensitivity comparisons. `MG_RAW_MANIFEST` can select another explicitly prepared manifest. Raw preprocessing requires the recorded OpenCV backend and stops if it is unavailable.

Processing outputs go to `MG_OUTPUT_DIR/upstream_imaging`. The short imaging runner reads the retained `coauthor_inputs/pipeline/` tree. It does not automatically switch to a newly processed raw-image run. A complete new processing run must be checked before its outputs replace the statistical inputs.

Use one numerical-library thread per worker and up to 14 workers. Check the memory demand of a representative stack before processing large stacks concurrently. Preserve detector settings, seeds and the number of spatial shifts.

## Morphology inputs

`morph1_topology_features.py` extracts topology measurements and branch barcodes from final label masks. It writes both files to `MG_OUTPUT_DIR/morphology/`.

`morph2_persistent_branches.py` prefers barcodes from the active run and otherwise reads the retained barcodes. It uses the final eligible-cell table for inclusion and metadata. The primary test recomputes the control percentile in every animal-label allocation. Fixed-threshold results are separate conditional sensitivities.

For the secondary measurements:

```sh
python code/analysis/morph3_secondary_features.py --topology-source deposited
```

To use a topology extraction from the active run:

```sh
python code/analysis/morph3_secondary_features.py --topology-source regenerated
```

The default `auto` mode selects the regenerated table when present. `--topology-features /path/to/table.csv` selects an explicit file. An explicitly requested file or regenerated table must exist. The script prints the chosen source and records its path and SHA256 hash. General shape measurements still come from the retained final feature table.

Primary, regional, threshold and secondary tests remain separate from the exploratory mixed-model screen. Run that screen with `--check-optimizers` to apply the recorded convergence checks under the same model specification.

## Other analyses

The analysis directory includes marker-ordering calibration, density multiplicity correction, distance and intensity checks, descriptive precision intervals and count-model sensitivities. Consult [CURRENT_CODE_SCOPE.md](CURRENT_CODE_SCOPE.md) for entry points and outputs.

`derived_input()` selects a regenerated predecessor in the active output directory when available and otherwise reads the retained input. Keep each run in a fresh output directory. Statistical assumptions require review independently of numerical reproduction.
