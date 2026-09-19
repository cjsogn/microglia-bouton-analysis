# Computational environments

These versions were inspected in the local environment used for the September 2026 reproduction checks. They are an execution record, not a claim that every upstream stage has been rerun or that packages are the latest releases.

## Python

The numerical reruns used Python 3.12.7. Create a separate environment and install only the dependency sets needed for the task:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
```

| Dependency set | Purpose |
| --- | --- |
| `requirements-core.txt` | Processed imaging, morphology and count-level runners, with tests |
| `requirements-single-cell.txt` | Raw transcriptomic preparation and annotation |
| `requirements-raw-image.txt` | Confocal preprocessing with the recorded OpenCV backend |
| `requirements-dev.txt` | Formatting and lint checks |

Core numerical versions were NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, statsmodels 0.14.5 and pyarrow 23.0.1. The single-cell environment contained Scanpy 1.11.5, AnnData 0.12.6, harmonypy 0.2.0, scikit-learn 1.9.0, numba 0.67.0 and llvmlite 0.49.0. Complete direct dependency pins are in the requirement files.

Some dependencies differ from earlier annotation environments. The retained annotation version records are in the data bundle. Current count-level reproduction uses those deposited selections. Repeating clustering is a separate task and requires marker review before reusing cluster identifiers. No claim of identical reclustering across dependency versions is made.

OpenCV 4.10.0 was the recorded local raw-image backend. The raw-image requirement selects `opencv-python-headless==4.10.0.84`. Install one OpenCV distribution, not overlapping headless and GUI variants. The raw loaders reject an unavailable OpenCV backend rather than substitute another morphological implementation. Installing this dependency was checked, but raw confocal processing was not repeated in the packaging pass.


## R

The count-level reruns used R 4.5.3 with edgeR 4.8.2, limma 3.66.0 and statmod 1.5.2. The raw importer additionally needs tximport 1.38.2 and rhdf5 2.54.1. Donor workbook extraction used readxl 1.4.5.

For a new compatible R installation:

```r
install.packages(c("BiocManager", "statmod", "readxl"))
BiocManager::install(c("edgeR", "limma", "tximport", "rhdf5"))
sessionInfo()
```

Check the resulting versions against this record. A current installer may resolve different versions. Do not replace a missing package with a different statistical method. Local compiled R libraries are not portable and are not included in the release.

## Parallel execution and reproducibility

Use up to 14 workers where the entry point supports them, with bounded batches. Set numerical-library threads to one per process to avoid multiplying worker and BLAS thread counts. Large confocal stacks need a representative memory check before concurrent processing. Preserve the recorded seeds, thresholds, models and number of spatial shifts.

Record package versions, commands, source checksums and output locations for each run. Keep inputs, local environments and outputs outside the GitHub code tree. The validation files in the full submission package contain the executed logs.
