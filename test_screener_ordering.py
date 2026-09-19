import pandas as pd
import pytest

from quantist_library.screener import (
    rank_flow_candidates,
    rank_vprofile_candidates,
    rank_vwap_candidates,
)


def test_money_flow_ranking_is_directional_and_code_deterministic():
    candidates = pd.DataFrame(
        {"mf": [10.0, 10.0, -20.0]},
        index=pd.Index(["zzzz", "aaaa", "mmmm"], name="code"),
    )

    assert rank_flow_candidates(candidates, n_stockcodes=3, ascending=False).index.tolist() == [
        "aaaa",
        "zzzz",
        "mmmm",
    ]
    assert rank_flow_candidates(candidates, n_stockcodes=3, ascending=True).index.tolist() == [
        "mmmm",
        "aaaa",
        "zzzz",
    ]


def test_vwap_rally_ranks_largest_positive_price_gap_first():
    data = pd.DataFrame(
        {"close": [101.0, 120.0], "vwap": [100.0, 100.0], "netval": [-100.0, 100.0]},
        index=pd.MultiIndex.from_tuples(
            [("near", 1), ("far", 1)],
            names=["code", "date"],
        ),
    )

    assert rank_vwap_candidates(data, ["near", "far"], 2, "vwap_rally") == ["far", "near"]


def test_vwap_around_ranks_closest_price_gap_first():
    data = pd.DataFrame(
        {"close": [101.0, 104.0], "vwap": [100.0, 100.0], "netval": [-100.0, 100.0]},
        index=pd.MultiIndex.from_tuples(
            [("near", 1), ("far", 1)],
            names=["code", "date"],
        ),
    )

    assert rank_vwap_candidates(data, ["near", "far"], 2, "vwap_around") == ["near", "far"]


def test_vwap_breakout_ranks_fresh_cross_before_larger_gap():
    data = pd.DataFrame(
        {"close": [90.0, 101.0, 110.0, 101.0], "vwap": [100.0] * 4, "netval": [1.0] * 4},
        index=pd.MultiIndex.from_tuples(
            [("fresh", 1), ("fresh", 2), ("stale", 1), ("stale", 2)],
            names=["code", "date"],
        ),
    )

    assert rank_vwap_candidates(data, ["fresh", "stale"], 2, "vwap_breakout") == ["fresh", "stale"]


def test_vwap_breakdown_ranks_fresh_cross_before_larger_gap():
    data = pd.DataFrame(
        {"close": [110.0, 99.0, 90.0, 99.0], "vwap": [100.0] * 4, "netval": [1.0] * 4},
        index=pd.MultiIndex.from_tuples(
            [("fresh", 1), ("fresh", 2), ("stale", 1), ("stale", 2)],
            names=["code", "date"],
        ),
    )

    assert rank_vwap_candidates(data, ["fresh", "stale"], 2, "vwap_breakdown") == ["fresh", "stale"]


def test_vprofile_ranking_uses_annotations_before_truncation_and_ties_by_code():
    candidates = pd.DataFrame(
        {
            "mf": [100.0, -50.0, 20.0, 20.0],
            "vprofile_zone_prominence": [0.30, 0.90, 0.90, 0.90],
            "vprofile_zone_strength": [0.90, 0.90, 0.80, 0.80],
        },
        index=pd.Index(["mf_first", "signal", "code_b", "code_a"], name="code"),
    )

    ranked = rank_vprofile_candidates(candidates, "vprofile_inside", 2)

    assert ranked.index.tolist() == ["signal", "code_a"]
    assert ranked["vprofile_zone_prominence"].tolist() == [0.90, 0.90]


@pytest.mark.parametrize(
    ("criteria", "expected"),
    [
        ("vprofile_breakout", ["high_mf", "low_mf"]),
        ("vprofile_support_bounce", ["high_mf", "low_mf"]),
        ("vprofile_breakdown", ["low_mf", "high_mf"]),
        ("vprofile_resistance_rejection", ["low_mf", "high_mf"]),
    ],
)
def test_vprofile_event_ranking_uses_directional_flow_tiebreak(criteria, expected):
    candidates = pd.DataFrame(
        {
            "mf": [100.0, -100.0],
            "vprofile_zone_prominence": [0.8, 0.8],
            "vprofile_zone_strength": [0.6, 0.6],
        },
        index=pd.Index(["high_mf", "low_mf"], name="code"),
    )

    assert rank_vprofile_candidates(candidates, criteria, 2).index.tolist() == expected
