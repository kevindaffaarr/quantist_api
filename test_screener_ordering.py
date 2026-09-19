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


@pytest.mark.parametrize("criteria", ["vwap_rally", "vwap_around", "vwap_breakout"])
def test_vwap_positive_criteria_rank_strongest_positive_flow_first(criteria):
    data = pd.DataFrame(
        {"netval": [4.0, 2.0, -1.0, 8.0]},
        index=pd.MultiIndex.from_tuples(
            [("low", 1), ("low", 2), ("high", 1), ("high", 2)],
            names=["code", "date"],
        ),
    )

    assert rank_vwap_candidates(data, ["low", "high"], 2, criteria) == ["high", "low"]


def test_vwap_breakdown_ranks_strongest_negative_flow_first():
    data = pd.DataFrame(
        {"netval": [-3.0, -7.0, -8.0, -9.0]},
        index=pd.MultiIndex.from_tuples(
            [("mild", 1), ("mild", 2), ("strong", 1), ("strong", 2)],
            names=["code", "date"],
        ),
    )

    assert rank_vwap_candidates(data, ["mild", "strong"], 2, "vwap_breakdown") == [
        "strong",
        "mild",
    ]


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
