# Morphology mask processing

This directory contains the processing steps used to obtain Iba1 cell masks:
background subtraction and Richardson-Lucy deconvolution, three-dimensional
hysteresis segmentation, soma-seeded watershed, and splitting of merged cells.
It is separate from the Iba1 compartment segmentation used for bouton association.

`parameters.py` contains the recorded acquisition and processing constants. They
are study-specific settings, not defaults validated for other microscopes or
specimens. The source mapping is in `../../docs/morphology_source_hashes.json`.

## Inputs and execution

Prepare a TSV or CSV manifest containing `field_id` and `path`. Each field
identifier must be unique. Paths may be absolute or relative to the manifest.
Use the deposited field identifiers to compare regenerated masks with retained
measurements. The raw-input loader follows the recorded ZYX or ZCYX layout,
selecting channel zero in a ZCYX acquisition. Confirm channel identity before
applying it to other images.

```sh
python code/morphology_processing/run_masks.py --manifest /path/to/input_manifest.tsv --out /path/to/new/mask_run --stage all --workers 14 --max-active-stacks 1
```

`--max-active-stacks` bounds concurrent volumes independently of the worker
request. Measure a representative run before increasing it. Use one numerical
library thread per worker. The runner refuses existing output directories,
duplicate inputs and output paths within the source image directory.

| Stage | Manifest inputs | Output subdirectory |
| --- | --- | --- |
| `preprocess` | Iba1 acquisition files | `deconvolved_stacks/` |
| `segment` | Deconvolved ZYX volumes | `initial_masks/` |
| `split` | Initial integer cell labels | `final_masks/` |
| `all` | Iba1 acquisition files | All three directories |

The runner records input and output hashes, software versions and parameters.
No input image is overwritten. A failed field stops the run. Existing partial
outputs are left available for inspection and are not silently treated as a
completed run.

## Method details

Segmentation uses the deconvolved volume without local contrast equalization.
Triangle and Otsu thresholds define three-dimensional hysteresis, followed by
small-component removal and closing. Physical distance transforms and smoothed
soma peaks seed watershed. Initial filtering applies the recorded volume limits
and excludes objects touching an axial acquisition face. Foreground is then
relabelled, so merged objects require the separate splitting stage.

Splitting identifies connected cores above the recorded inscribed-radius and
core-volume thresholds. Each core's distance maximum becomes a watershed seed.
Subregions failing the volume or radius rule are absorbed into the largest
remaining subregion. This rule is reproduced as recorded, without replacing it
with a nearest-neighbour assignment.

## Relationship to the retained analyses

Mask generation does not determine the complete eligible-cell analysis set.
The retained general feature table additionally applies volume and radius
criteria. The downstream statistics continue to use that explicit table and
its cell identifiers. A new mask run must be reconciled with those identifiers
before statistical reuse.

The retained mask files were matched byte for byte to their original processing
outputs. That source reconciliation is distinct from repeating preprocessing of
every acquisition. See `../../docs/VALIDATION.md` for executed checks. The complete
general-feature extraction workflow is not part of this mask-generation command.
