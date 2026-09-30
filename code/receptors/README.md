# Moon microglial receptor and temporal profiles

Figure 7B contains the same gene panel as the replicated mouse and human analyses. Figure 7C contains additional receptors, without duplicating genes from 7B. Dot area represents detection prevalence and colour represents the descriptive change relative to PBS. Figure S4 reports absolute prevalence and mean log1p expression.

The condition libraries support descriptive profiles, not treatment tests with cells as independent replicates.

## Processing sequence

From the code root, supply the source HDF5 files and retained selection reference:

```sh
python code/receptors/04_moon_timecourse.py --raw-dir /path/to/GSE307796 --reference-table /path/to/coauthor_inputs/external/receptor_prevalence_with_counts.csv --panel /path/to/coauthor_inputs/external/moon_timecourse_inputs/curated_gene_panel.tsv --out /path/to/new/moon_profile
python code/receptors/05_moon_selection_diagnostics.py --data-dir /path/to/new/moon_profile --out /path/to/new/moon_selection_checks
python code/receptors/06_moon_descriptive_ratios.py --data /path/to/new/moon_profile --out /path/to/new/moon_ratios/expression_prevalence.tsv
```

Scripts 05 and 06 can also read the retained `03_data/corrected/moon_timecourse/` directory. Outputs must remain outside the input bundle.

Script 04 reproduces the recorded cell selection, verifies the baseline receptor summaries and exports selected-cell raw counts, full-feature summaries and selection diagnostics. Prevalence uses nonzero raw counts. Mean log1p expression includes zero-count cells after normalization.

Script 05 evaluates the stated identity-marker restrictions without replacing the primary annotation.

Script 06 calculates arithmetic mean per-cell normalized counts before log transformation, then log2 ratios to PBS without a pseudocount. It reads the selected-cell count matrix in bounded batches.

## Selection definitions

`selection_and_receptor_catalogue.json` contains the microglial and competing-lineage markers, neurotransmitter-family mapping and receptor catalogue. Selection uses the recorded gene-filtered expression matrix. Expression summaries use the full shared raw feature universe. These are distinct processing steps, not competing analyses.

The reporting threshold is not an assay detection limit. Catalogue annotations do not establish receptor function. See [validation](../../docs/VALIDATION.md).
