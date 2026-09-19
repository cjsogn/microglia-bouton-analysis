# -*- coding: utf-8 -*-
"""Shared primitives for the marker-exchangeability calibration.

During development these lived in a module written into a working cache that was not retained,
so `baseline_ordering_calibration_variants.py` and `baseline_ordering_pairwise_calibration.py`
could not be run from the release. Every symbol they need is defined by the primary calibration
script, which is deposited, so this module re-exports them rather than reimplementing anything.
There is exactly one implementation of the test, and it is the one that produced the reported
calibration.
"""

from baseline_ordering_calibration import (  # noqa: F401
    ANIMALS,
    COL,
    KEY,
    MARKERS,
    NPERM,
    perm_index,
    rank_matrix,
    run_test,
)
