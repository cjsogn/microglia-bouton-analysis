# -*- coding: utf-8 -*-
"""Variant-simulation primitives, re-exported for the pairwise calibration.

Companion to `exch_sim`: the pairwise script imports the observed P values and the bank-loading
and scaling helpers from the variants module, which defines them. Re-exported here under the
name the pairwise script expects, so there is one implementation rather than a copy.
"""

from baseline_ordering_calibration_variants import (  # noqa: F401
    NPERM,
    OBS_P,
    load,
    prep,
)
