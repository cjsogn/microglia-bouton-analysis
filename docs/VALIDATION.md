# Validation scope, 19 September 2026

Core imaging summaries, nine mouse RNA contrasts, two mouse cell-selection checks,
four human SNpc fits and thirteen putamen fits were rerun from retained processed
inputs in the relocated package. Thirty-three output tables, including complete
gene-level results, matched with relative tolerance 1e-8 and absolute tolerance
1e-10. The primary morphology estimates also matched their retained numerical
columns. Secondary morphology and regional temporal analyses were rerun directly
from this analysis-only repository. The extracted repository passed all 24 tests
and the configured static checks.

The recovered exploratory morphology screen was rerun with explicit convergence
diagnostics. Three default comparisons had failed convergence. Retrying the same
maximum-likelihood models with Powell and BFGS produced converged fits. Both the
original and corrected probabilities and all warnings are retained in the separate
validation bundle. Use `--check-optimizers` to apply this numerical correction.

The current pass did not rerun raw-confocal segmentation, the full spatial null,
or every upstream RNA annotation stage. Their code and retained inputs are supplied
where available. These limits are not a claim that the upstream stages were newly
validated end to end. See `ANALYSIS_SCOPE.md` before reproducing longer stages.
