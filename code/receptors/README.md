# Moon microglial receptor and temporal profiles

Figure 7d uses the same gene order as the mouse panels and their human counterparts. Figure 7e is a receptor dot plot, excluding genes already displayed in 7d. Dot area represents detection prevalence. Color represents the descriptive change relative to PBS. Figure S5 retains absolute prevalence and mean log1p expression.

These libraries provide descriptive condition profiles. Cells are not independent treatment replicates, so the pipeline does not calculate treatment P values for them.

## Current pipeline

From the code root, supply the three original HDF5 files and the retained selection reference:

```sh
python code/receptors/04_moon_timecourse.py --raw-dir /path/to/GSE307796 --reference-table /path/to/coauthor_inputs/external/receptor_prevalence_with_counts.csv --panel /path/to/coauthor_inputs/external/moon_timecourse_inputs/curated_gene_panel.tsv --out /path/to/new/moon_profile
python code/receptors/05_moon_selection_diagnostics.py --data-dir /path/to/new/moon_profile --out /path/to/new/moon_selection_checks
python code/receptors/06_moon_descriptive_ratios.py --data /path/to/new/moon_profile --out /path/to/new/moon_ratios/expression_prevalence.tsv
```

Outputs are separate from inputs. Scripts 05 and 06 also accept the deposited `03_data/corrected/moon_timecourse/` directory as input.

Script 04 reproduces the original cell selection, checks the earlier receptor summaries, and exports the full shared raw feature universe, selected-cell raw counts, selection diagnostics and gene-filter audit. Prevalence uses nonzero raw counts. Mean log1p expression includes zeros after per-cell normalization. The diagnostic restrictions do not replace the primary selection.

Script 06 computes arithmetic mean per-cell normalized counts before log transformation and then descriptive log2 ratios to PBS, without a pseudocount. It reads the raw selected-cell matrix in bounded batches. Its augmented table supplies the current temporal heatmap and receptor colors.

For the deposited current figures:

```sh
```


## Selection and receptor definitions

`selection_and_receptor_catalogue.json` contains the unchanged microglial and non-microglial marker lists, neurotransmitter-family mapping and broader receptor catalogue. These definitions were extracted exactly from the earlier scripts. Stage 04 reads this configuration directly, without depending on superseded analysis scripts.

The current full-feature summaries reproduce the retained baseline prevalence. The reporting threshold is not an assay detection limit. Biological tags are catalogue metadata, not evidence of receptor function. Earlier table assembly and unused figure versions are outside the release.

See [validation](../../docs/VALIDATION.md).
