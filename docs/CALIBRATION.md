# Baseline-ordering calibration

These simulations assess the behavior of exploratory marker-ordering tests. They are separate from the short reproduction entry points. See [VALIDATION.md](VALIDATION.md) for executed checks.

The original homogeneous calibration uses seed 424242 and a nominal request of 20,000 simulations. Variant simulations use seed 909090 unless `SEED2` is explicitly supplied. The deposited observed and matched variants used a nominal request of 100,000. Pairwise calibration uses seed 771177 and a nominal request of 20,000. Read the active source before launching and retain its random streams.

Parallel chunking takes the integer quotient of the requested simulations across 14 workers. A request of 20,000 therefore realizes 19,992 draws, and 100,000 realizes 99,988. The recorded allocation and random streams are retained for reproduction. Report the exported `n_sim`, which is the completed count.

The assembler requires all four scheme files in the output `_cache/` directory:

- `typeI_homog.tsv`
- `typeI_matched.tsv`
- `typeI_observed.tsv`
- `typeI_sampling.tsv`

It also requires `marker_null_variance.tsv` and an explicitly selected `typeI_pairs_MODE.tsv` file. Each scheme must contain eight unique endpoint/test rows. Pairwise output must contain twenty rows, with five comparator tests in each endpoint/scale family. BH adjustment is applied separately within those families. Missing or malformed inputs fail before assembled outputs are written.

```sh
python code/analysis/assemble_ordering_calibration.py --pairwise-mode sampling
```

Select the pairwise mode from the run records before regeneration. The example selects `sampling`, but does not establish which mode produced a particular retained table. Do not mix cached schemes from different parameter settings. The complete retained calibration tables remain the reported source until a documented rerun replaces them.

The simulations contain nominal test values for the retained observations. Recompute and verify these inputs before applying the calibration to a different dataset. Load pickle caches only from trusted runs.
