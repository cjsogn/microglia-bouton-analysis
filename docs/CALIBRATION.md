# Baseline-ordering calibration

These simulations assess the behavior of exploratory marker-ordering tests. They are not part of the short reproduction entry points and were not all rerun during this completion pass.

The original homogeneous calibration uses seed 424242 and a nominal request of 20,000 simulations. Variant simulations use seed 909090 unless `SEED2` is explicitly supplied. The deposited observed and matched variants used a nominal request of 100,000. Pairwise calibration uses seed 771177 and a nominal request of 20,000. Read the active source before launching and retain its random streams.

Historical parallel chunking takes the integer quotient of the requested simulations across 14 workers. Thus a request of 20,000 realizes 19,992 draws, and 100,000 realizes 99,988. This behavior is retained to avoid changing deposited simulation outputs. Exported `n_sim`, not the requested environment value, is the realized count. Do not claim 20,000 or 100,000 completed draws when describing those exact runs.

The assembler requires all four scheme files in the output `_cache/` directory:

- `typeI_homog.tsv`
- `typeI_matched.tsv`
- `typeI_observed.tsv`
- `typeI_sampling.tsv`

It also requires `marker_null_variance.tsv` and an explicitly selected `typeI_pairs_MODE.tsv` file. Each scheme must contain eight unique endpoint/test rows. Pairwise output must contain twenty rows, with five comparator tests in each endpoint/scale family. BH adjustment is applied separately within those families. Missing or malformed inputs fail before assembled outputs are written.

```sh
python code/analysis/assemble_ordering_calibration.py --pairwise-mode sampling
```

That command is an example of an explicit choice, not a claim that all historical pairwise-cache provenance has been recovered. Confirm the intended simulation mode from the run records before regeneration. Do not mix cached schemes from different parameter settings. The deposited complete calibration tables remain the source for the manuscript unless a fully documented rerun replaces them.

The simulations contain historical nominal test values used to calibrate those particular observations. They are not a generic analysis engine for newly changed data. Recompute and verify the nominal inputs before applying the calibration to a different dataset. Local pickle caches must only be loaded from trusted runs, never from untrusted external files.
