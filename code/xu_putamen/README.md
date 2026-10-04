# Human putamen microglia, GSE161045

This pipeline produces the putamen column to the right of the SNpc column in the human PD analysis. It uses independent microglial annotation and donor pseudobulk. The two human cohorts are fitted separately and do not alter mouse FDR families.

The input bundle is `03_data/xu_putamen/`. It includes raw files, retained annotations, donor counts, differential-expression results and provenance. No raw counts or generated results belong in the code-only repository.

## Short donor-level rerun

From the code root:

```sh
python scripts/reproduce_human_from_counts.py
```

This fits both human cohorts using their deposited selections. It does not repeat putamen annotation.

## Independent annotation

Use a separate workspace. The preparation command copies raw inputs into it so annotation checkpoints cannot overwrite the deposited bundle.

```sh
python scripts/prepare_putamen_workspace.py --mode annotation --out /path/to/new/putamen_workspace
export MG_XU_WORK_DIR="/path/to/new/putamen_workspace"
python code/xu_putamen/code/02_audit_source_counts.py
Rscript code/xu_putamen/code/03_extract_donor_metadata.R /path/to/new/putamen_workspace/raw/supplementary_tables_springer.xlsx /path/to/new/putamen_workspace/results/published_donor_metadata.csv
python code/xu_putamen/code/04_independent_annotation.py prepare
```

Inspect lineage markers and clustering before selecting immune clusters. The retained run used `immune --clusters 11 12`, followed by `finalize --clusters 2 3 4 5 6 7 8 --harmony-clusters 1 2 4 7 8`. Those identifiers describe the recorded run, not guaranteed identities after reclustering. The fixed eleven-gene list `ANNOTATION_EXCLUSIONS` is excluded from clustering features and annotation marker rankings. It does not include ADRB2 and is distinct from the 25-gene reporting catalogue. Harmony affects only the alternative annotation, not the raw counts used for testing.

After annotation:

```sh
Rscript code/xu_putamen/code/05_donor_pseudobulk.R /path/to/new/putamen_workspace/results/independent_annotation /path/to/new/putamen_workspace/results/differential_expression
python code/xu_putamen/code/06_verify_analysis.py
python code/xu_putamen/code/08_validate_numeric_backend.py
```

The model input directory must contain the primary, harmony and concordant donor-count subdirectories. The result directory must be fresh and outside that input directory.

The primary donor model includes age, sex and diagnosis. The retained sensitivities cover PMI adjustment, a diagnosis-only model, alternative and concordant annotations, and donor omissions. Each fit has its own expression filter and whole-transcriptome BH family. An untested panel gene remains untested, not a nonsignificant numerical estimate.
