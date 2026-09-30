# -*- coding: utf-8 -*-
"""Variant-simulation primitives, re-exported for the pairwise calibration.

The pairwise simulation uses the bank-loading and scaling helpers defined by
the variance-scheme simulation.
"""

from baseline_ordering_calibration_variants import (  # noqa: F401
    NPERM,
    OBS_P,
    load,
    prep,
)
