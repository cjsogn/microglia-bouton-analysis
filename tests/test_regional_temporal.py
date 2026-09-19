"""Synthetic fixtures test software behaviour, not experimental values."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "code/analysis"),
    str(Path(__file__).resolve().parents[1] / "code"),
]
from regional_temporal_contrast import (
    temporal_profile_test,
    bootstrap_summary,
    regional_marker_summaries,
)


def test_profile_test_invariant_to_animal_main_effect_and_region_order():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(9, 5))
    groups = np.arange(9) >= 5
    original = temporal_profile_test(x, groups)
    shifted = temporal_profile_test(x + np.arange(9)[:, None], groups)
    reverse = temporal_profile_test(x[:, ::-1], groups)
    assert original["n_allocations"] == 126
    assert original["exact_p"] == shifted["exact_p"] == reverse["exact_p"]
    np.testing.assert_allclose(original["statistic"], shifted["statistic"])


def test_common_time_shift_is_not_interaction():
    x = np.tile(np.arange(5, dtype=float), (9, 1))
    groups = np.arange(9) >= 5
    x[groups] += 2
    result = temporal_profile_test(x, groups)
    assert result["statistic"] == 0 and result["exact_p"] == 1


def test_incomplete_profiles_fail():
    x = np.ones((9, 5))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="Complete finite"):
        temporal_profile_test(x, np.arange(9) >= 5)


def test_bootstrap_is_exhaustive_and_matches_mean():
    result = bootstrap_summary([1, 2, 3, 4, 5])
    assert result["n_bootstrap_resamples"] == 3125
    assert result["estimate"] == 3
    result = bootstrap_summary([1, 2, 4], log=True)
    np.testing.assert_allclose(result["estimate"], 2)


def test_regional_summary_keeps_condition_and_animal_units_separate():
    animals = pd.DataFrame(
        dict(
            animal_id=["a", "b", "c", "d", "e"],
            condition=["Control"] * 3 + ["LPS_60d"] * 2,
            region=["CA1"] * 5,
            marker=["TH"] * 5,
            enrichment=[1, 2, 3, 5, 7],
        )
    )
    result = regional_marker_summaries(animals).set_index("condition")
    assert result.loc["Control", "estimate"] == 2
    assert result.loc["LPS_60d", "estimate"] == 6
    assert result.n_animals.to_dict() == {"Control": 3, "LPS_60d": 2}
    assert result.n_bootstrap_resamples.to_dict() == {"Control": 27, "LPS_60d": 4}
    with pytest.raises(ValueError, match="One row per animal"):
        regional_marker_summaries(pd.concat([animals, animals.iloc[:1]]))


@pytest.mark.parametrize("values", [[], [1], [1, np.nan], [1, np.inf]])
def test_bootstrap_rejects_incomplete_input(values):
    with pytest.raises(ValueError, match="finite animal values"):
        bootstrap_summary(values)


def test_log_bootstrap_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="positive"):
        bootstrap_summary([0, 1], log=True)
